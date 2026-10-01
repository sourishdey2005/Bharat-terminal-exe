// crates/bt-data/src/nse.rs
// Author: Sourish Dey

//! NSE India provider for F&O chains, Bhavcopy, and indices.
//!
//! Uses a cookie jar: first hits https://www.nseindia.com to obtain
//! session cookies, then queries API endpoints.

use bt_core::{BtError, Result};
use reqwest::cookie::Jar;
use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";
const BASE_URL: &str = "https://www.nseindia.com";

#[derive(Debug, Clone, Deserialize)]
pub struct NSEOptionChain {
    pub strike: f64,
    pub call_oi: u64,
    pub call_ltp: f64,
    pub call_iv: f64,
    pub put_oi: u64,
    pub put_ltp: f64,
    pub put_iv: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NSEIndex {
    pub name: String,
    pub last: f64,
    pub change: f64,
    pub change_pct: f64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NSEOptionChainRecord {
    #[serde(rename = "strikePrice")]
    strike_price: f64,
    #[serde(rename = "CE")]
    ce: Option<NSEOptionData>,
    #[serde(rename = "PE")]
    pe: Option<NSEOptionData>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NSEOptionData {
    #[serde(rename = "openInterest")]
    open_interest: u64,
    #[serde(rename = "lastPrice")]
    last_price: f64,
    #[serde(rename = "impliedVolatility")]
    implied_volatility: f64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NSEOptionChainResponse {
    records: NSEOptionChainRecords,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NSEOptionChainRecords {
    data: Vec<NSEOptionChainRecord>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NSEIndexItem {
    #[serde(rename = "symbol")]
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: f64,
    #[serde(rename = "change")]
    change: f64,
    #[serde(rename = "pChange")]
    p_change: f64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NSEIndexResponse {
    data: Vec<NSEIndexItem>,
}

pub struct NSEProvider {
    client: Client,
}

impl NSEProvider {
    pub fn new() -> Self {
        let jar = Jar::default();
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .cookie_provider(Arc::new(jar))
            .timeout(StdDuration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    /// Establish a session by hitting the NSE homepage to get cookies.
    #[instrument(skip(self))]
    pub async fn init_session(&self) -> Result<()> {
        let resp = self
            .client
            .get(BASE_URL)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("NSE session init failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(BtError::DataFetch(format!(
                "NSE session init HTTP {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Fetch option chain for an index (e.g., "NIFTY", "BANKNIFTY").
    #[instrument(skip(self))]
    pub async fn fetch_option_chain(&self, symbol: &str) -> Result<Vec<NSEOptionChain>> {
        let url = format!(
            "{}/api/option-chain-indices?symbol={}",
            BASE_URL,
            urlencoding::encode(symbol)
        );
        let resp = self.fetch_with_retry(&url).await?;
        let data: NSEOptionChainResponse = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("JSON parse error: {}", e)))?;

        let chains: Vec<NSEOptionChain> = data
            .records
            .data
            .into_iter()
            .map(|r| NSEOptionChain {
                strike: r.strike_price,
                call_oi: r.ce.as_ref().map(|c| c.open_interest).unwrap_or(0),
                call_ltp: r.ce.as_ref().map(|c| c.last_price).unwrap_or(0.0),
                call_iv: r.ce.as_ref().map(|c| c.implied_volatility).unwrap_or(0.0),
                put_oi: r.pe.as_ref().map(|p| p.open_interest).unwrap_or(0),
                put_ltp: r.pe.as_ref().map(|p| p.last_price).unwrap_or(0.0),
                put_iv: r.pe.as_ref().map(|p| p.implied_volatility).unwrap_or(0.0),
            })
            .collect();

        Ok(chains)
    }

    /// Fetch index constituents (e.g., "NIFTY 50").
    #[instrument(skip(self))]
    pub async fn fetch_index(&self, index: &str) -> Result<Vec<NSEIndex>> {
        let url = format!(
            "{}/api/equity-stockIndices?index={}",
            BASE_URL,
            urlencoding::encode(index)
        );
        let resp = self.fetch_with_retry(&url).await?;
        let data: NSEIndexResponse = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("JSON parse error: {}", e)))?;

        let indices: Vec<NSEIndex> = data
            .data
            .into_iter()
            .map(|item| NSEIndex {
                name: item.symbol,
                last: item.last_price,
                change: item.change,
                change_pct: item.p_change,
            })
            .collect();

        Ok(indices)
    }

    async fn fetch_with_retry(&self, url: &str) -> Result<reqwest::Response> {
        let mut attempts = 0;
        const MAX_RETRIES: u32 = 3;

        loop {
            let resp = self
                .client
                .get(url)
                .send()
                .await
                .map_err(|e| BtError::DataFetch(format!("Network error: {}", e)))?;

            if resp.status() == 429 || resp.status() == 401 {
                attempts += 1;
                if attempts >= MAX_RETRIES {
                    return Err(BtError::DataFetch(format!(
                        "Rate limited ({}) after {} retries",
                        resp.status(),
                        MAX_RETRIES
                    )));
                }
                let delay = StdDuration::from_millis(500 * 2_u64.pow(attempts - 1));
                tokio::time::sleep(delay).await;
                continue;
            }

            if !resp.status().is_success() {
                return Err(BtError::DataFetch(format!(
                    "HTTP {}: {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                )));
            }

            return Ok(resp);
        }
    }
}

impl Default for NSEProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// One day of institutional flow. Values are ₹ crore, positive = net buy.
#[derive(Debug, Clone, PartialEq)]
pub struct FiiDiiActivity {
    pub date: String,
    pub fii_net: f64,
    pub dii_net: f64,
}

/// One India VIX daily bar.
#[derive(Debug, Clone, PartialEq)]
pub struct VixBar {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

impl NSEProvider {
    /// Daily FII/DII net activity (₹ crore).
    ///
    /// NSE rotates these auxiliary paths without notice, so the endpoint lives
    /// in one constant and parsing is lenient across field spellings. A moved or
    /// blocked endpoint fails closed as `Err`, never as a panic or empty vec
    /// that callers could mistake for "no flow".
    pub async fn fetch_fii_dii_activity(&self) -> Result<Vec<FiiDiiActivity>> {
        const URL: &str = "https://www.nseindia.com/api/fii-dii-trading-activity";
        let resp = self.fetch_with_retry(URL).await?;
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("FII/DII JSON error: {e}")))?;
        parse_fii_dii_body(&body)
    }

    /// India VIX daily history between `from` and `to` (`DD-MM-YYYY`).
    pub async fn fetch_daily_volatility(&self, from: &str, to: &str) -> Result<Vec<VixBar>> {
        let url = format!("{BASE_URL}/api/historical/vixhistory?from={from}&to={to}");
        let resp = self.fetch_with_retry(&url).await?;
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("VIX JSON error: {e}")))?;
        parse_vix_body(&body)
    }

    /// Top `n` gainers of an index (e.g. `"NIFTY 50"`) by day change %.
    ///
    /// Reuses the same `equity-stockIndices` endpoint as [`Self::fetch_index`],
    /// so no new endpoint is introduced; the ranking is done client-side.
    pub async fn fetch_top_gainers(&self, index: &str, n: usize) -> Result<Vec<NSEIndex>> {
        let mut rows = self.fetch_index(index).await?;
        rows.sort_by(|a, b| {
            b.change_pct
                .partial_cmp(&a.change_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows.truncate(n.max(1));
        Ok(rows)
    }

    /// Top `n` losers of an index by day change %.
    pub async fn fetch_top_losers(&self, index: &str, n: usize) -> Result<Vec<NSEIndex>> {
        let mut rows = self.fetch_index(index).await?;
        rows.sort_by(|a, b| {
            a.change_pct
                .partial_cmp(&b.change_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows.truncate(n.max(1));
        Ok(rows)
    }

    /// Constituents sorted by absolute day change % (a volume-free proxy for
    /// "most active" from the same endpoint; true traded volumes need the full
    /// quote feed).
    pub async fn fetch_most_active(&self, index: &str, n: usize) -> Result<Vec<NSEIndex>> {
        let mut rows = self.fetch_index(index).await?;
        rows.sort_by(|a, b| {
            b.change_pct
                .abs()
                .partial_cmp(&a.change_pct.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows.truncate(n.max(1));
        Ok(rows)
    }
}

/// Lenient FII/DII parsing across the field spellings NSE has used.
pub fn parse_fii_dii_body(body: &serde_json::Value) -> Result<Vec<FiiDiiActivity>> {
    let rows: &[serde_json::Value] = body
        .get("data")
        .and_then(|v| v.as_array())
        .map(Vec::as_slice)
        .or_else(|| body.as_array().map(Vec::as_slice))
        .ok_or_else(|| BtError::InvalidInput("FII/DII: no data array".into()))?;
    let mut out = Vec::new();
    for r in rows {
        let obj = match r.as_object() {
            Some(o) => o,
            None => continue,
        };
        let get = |keys: &[&str]| -> Option<f64> {
            keys.iter().find_map(|k| {
                obj.iter().find_map(|(name, v)| {
                    if name.to_lowercase().contains(&k.to_lowercase()) {
                        v.as_f64().or_else(|| {
                            v.as_str()
                                .and_then(|s| s.replace(',', "").parse::<f64>().ok())
                        })
                    } else {
                        None
                    }
                })
            })
        };
        let date = obj
            .iter()
            .find(|(k, _)| k.to_lowercase().contains("date"))
            .and_then(|(_, v)| v.as_str())
            .unwrap_or("")
            .to_string();
        let (Some(fii), Some(dii)) = (
            get(&["fiinet", "fii_net", "fiin et"]),
            get(&["diinet", "dii_net"]),
        ) else {
            continue;
        };
        out.push(FiiDiiActivity {
            date,
            fii_net: fii,
            dii_net: dii,
        });
    }
    Ok(out)
}

/// Lenient India VIX parsing (`DATE/OPEN/HIGH/LOW/CLOSE`, case-insensitive).
pub fn parse_vix_body(body: &serde_json::Value) -> Result<Vec<VixBar>> {
    let rows = body
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| BtError::InvalidInput("VIX: no data array".into()))?;
    let mut out = Vec::new();
    for r in rows {
        let obj = match r.as_object() {
            Some(o) => o,
            None => continue,
        };
        let get = |key: &str| -> Option<f64> {
            obj.iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(key))
                .and_then(|(_, v)| {
                    v.as_f64().or_else(|| {
                        v.as_str()
                            .and_then(|s| s.replace(',', "").parse::<f64>().ok())
                    })
                })
        };
        let date = obj
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("date"))
            .and_then(|(_, v)| v.as_str())
            .unwrap_or("")
            .to_string();
        let (Some(o), Some(h), Some(l), Some(c)) =
            (get("open"), get("high"), get("low"), get("close"))
        else {
            continue;
        };
        out.push(VixBar {
            date,
            open: o,
            high: h,
            low: l,
            close: c,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live NSE session test. Requires network access to nseindia.com.
    /// Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to nseindia.com"]
    async fn test_init_session() {
        let provider = NSEProvider::new();
        let result = provider.init_session().await;
        assert!(result.is_ok());
    }

    /// Live NSE option-chain test. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to nseindia.com"]
    async fn test_fetch_option_chain() {
        let provider = NSEProvider::new();
        provider.init_session().await.unwrap();
        let chain = provider.fetch_option_chain("NIFTY").await;
        assert!(chain.is_ok());
        assert!(!chain.unwrap().is_empty());
    }

    /// Live NSE index quote test. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to nseindia.com"]
    async fn test_fetch_index() {
        let provider = NSEProvider::new();
        provider.init_session().await.unwrap();
        let index = provider.fetch_index("NIFTY 50").await;
        assert!(index.is_ok());
        assert!(!index.unwrap().is_empty());
    }

    #[test]
    fn fii_dii_parsing_accepts_field_variants() {
        let body = serde_json::json!({"data": [
            {"date": "30-Sep-2026", "fiiNet": 1234.5, "diiNet": -321.0},
            {"TRADEDATE": "29-Sep-2026", "FII_NET": "2,000.0", "DII_NET": "500"}
        ]});
        let out = parse_fii_dii_body(&body).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].fii_net, 1234.5);
        assert_eq!(out[1].dii_net, 500.0);
    }

    #[test]
    fn fii_dii_skips_rows_without_both_legs() {
        let body = serde_json::json!({"data": [{"date": "x", "fiiNet": 1.0}]});
        assert!(parse_fii_dii_body(&body).unwrap().is_empty());
        assert!(parse_fii_dii_body(&serde_json::json!({})).is_err());
    }

    #[test]
    fn vix_parsing_accepts_case_variants() {
        let body = serde_json::json!({"data": [
            {"DATE": "30-Sep-2026", "OPEN": 12.1, "HIGH": 12.8, "LOW": 11.9, "CLOSE": 12.5}
        ]});
        let out = parse_vix_body(&body).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].close, 12.5);
        assert_eq!(out[0].date, "30-Sep-2026");
    }
}

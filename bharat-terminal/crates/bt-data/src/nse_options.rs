// crates/bt-data/src/nse_options.rs
// Author: Sourish Dey

//! NSE India option chains (no API key).
//!
//! Mirrors the session discipline in [`crate::nse`]: warm up with the
//! option-chain landing page to obtain cookies, then query the JSON API with
//! browser headers. NSE rate-limits aggressively (403), so every fetch retries
//! with backoff and fails closed with a message that names the remedy instead
//! of an opaque status code.
//!
//! Two chain families exist and the symbol decides which one is used:
//! indices (`NIFTY`, `BANKNIFTY`, `FINNIFTY`) go to `option-chain-indices`,
//! everything else to `option-chain-equities`. Expiry lists come from
//! `option-chain-contract-info`.

use bt_core::{BtError, OptionChain, OptionLeg, OptionStrike, Result};
use reqwest::cookie::Jar;
use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration as StdDuration, SystemTime, UNIX_EPOCH};
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";
const BASE_URL: &str = "https://www.nseindia.com";
const WARMUP_URL: &str = "https://www.nseindia.com/option-chain";

/// Index underlyings served by the indices endpoint.
pub const INDEX_SYMBOLS: &[&str] = &["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"];

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChainLeg {
    #[serde(rename = "openInterest", default)]
    open_interest: u64,
    #[serde(rename = "changeinOpenInterest", default)]
    change_oi: i64,
    #[serde(rename = "totalTradedVolume", default)]
    volume: u64,
    #[serde(rename = "lastPrice", default)]
    last_price: f64,
    #[serde(rename = "impliedVolatility", default)]
    iv: f64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChainRecord {
    #[serde(rename = "strikePrice")]
    strike_price: f64,
    #[serde(rename = "expiryDate")]
    expiry_date: String,
    #[serde(rename = "CE", default)]
    ce: Option<ChainLeg>,
    #[serde(rename = "PE", default)]
    pe: Option<ChainLeg>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ChainResponse {
    records: ChainRecords,
    #[serde(rename = "filtered")]
    filtered: ChainFiltered,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChainRecords {
    data: Vec<ChainRecord>,
    #[serde(rename = "expiryDates")]
    expiry_dates: Vec<String>,
    #[serde(rename = "underlyingValue")]
    underlying_value: Option<f64>,
    timestamp: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChainFiltered {
    data: Vec<ChainRecord>,
}

pub struct NseOptionsProvider {
    client: Client,
}

impl NseOptionsProvider {
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

    /// Warm up the cookie session via the option-chain landing page.
    #[instrument(skip(self))]
    pub async fn init_session(&self) -> Result<()> {
        let resp = self
            .client
            .get(WARMUP_URL)
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "en-US,en;q=0.9")
            .header("Referer", BASE_URL)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("NSE options warmup failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(BtError::DataFetch(format!(
                "NSE blocked (HTTP {}) — try again in 30s",
                resp.status()
            )));
        }
        Ok(())
    }

    fn chain_url(symbol: &str, expiry: Option<&str>) -> String {
        let upper = symbol.to_uppercase();
        let is_index = INDEX_SYMBOLS.contains(&upper.as_str());
        let base = if is_index {
            "option-chain-indices"
        } else {
            "option-chain-equities"
        };
        let mut url = format!("{BASE_URL}/api/{base}?symbol={upper}");
        if let Some(e) = expiry {
            url.push_str(&format!("&expiryDate={}", urlencoding::encode(e)));
        }
        url
    }

    async fn get_json(&self, url: &str) -> Result<ChainResponse> {
        let mut attempts = 0u32;
        loop {
            let resp = self
                .client
                .get(url)
                .header("Accept", "application/json")
                .header("Accept-Language", "en-US,en;q=0.9")
                .header("Referer", "https://www.nseindia.com/option-chain")
                .send()
                .await
                .map_err(|e| BtError::DataFetch(format!("NSE options network error: {e}")))?;
            let status = resp.status();
            if status == 403 || status == 429 {
                attempts += 1;
                if attempts >= 3 {
                    return Err(BtError::DataFetch(
                        "NSE blocked — try again in 30s".to_string(),
                    ));
                }
                tokio::time::sleep(StdDuration::from_secs(2 * attempts as u64)).await;
                continue;
            }
            if !status.is_success() {
                return Err(BtError::DataFetch(format!("NSE options HTTP {status}")));
            }
            return resp.json().await.map_err(|e| {
                BtError::InvalidInput(format!("NSE options JSON error: {e}"))
            });
        }
    }

    /// All listed expiries for a symbol, nearest first as published.
    #[instrument(skip(self))]
    pub async fn fetch_expiries(&self, symbol: &str) -> Result<Vec<String>> {
        let body = self.get_json(&Self::chain_url(symbol, None)).await?;
        Ok(body.records.expiry_dates)
    }

    /// Full chain for a symbol, optionally pinned to one expiry date string
    /// exactly as listed by [`Self::fetch_expiries`] (e.g. `"30-Oct-2026"`).
    #[instrument(skip(self))]
    pub async fn fetch_chain(
        &self,
        symbol: &str,
        expiry: Option<&str>,
    ) -> Result<OptionChain> {
        let body = self
            .get_json(&Self::chain_url(symbol, expiry))
            .await?;
        build_chain(symbol, expiry, &body)
    }

    /// Convenience: nearest-expiry NIFTY chain.
    pub async fn fetch_nifty_chain(&self) -> Result<OptionChain> {
        self.fetch_chain("NIFTY", None).await
    }

    /// Convenience: nearest-expiry BANKNIFTY chain.
    pub async fn fetch_banknifty_chain(&self) -> Result<OptionChain> {
        self.fetch_chain("BANKNIFTY", None).await
    }
}

impl Default for NseOptionsProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// `true` when the symbol has listed equity/index options on NSE.
///
/// This is a ticket check, not a fetch: NSE lists F&O on ~180 underlyings, so
/// anything outside the index list plus a curated equity set reports no
/// options rather than burning a rate-limited request to find out.
pub fn is_fno_symbol(symbol: &str) -> bool {
    let upper = symbol.to_uppercase();
    let base = upper
        .strip_suffix(".NS")
        .or_else(|| upper.strip_suffix(".BO"))
        .unwrap_or(&upper);
    if INDEX_SYMBOLS.contains(&base) {
        return true;
    }
    // Nifty-50 and large-cap names that carry listed derivatives. Kept as an
    // explicit set so an unknown ticker fails fast instead of 404-chasing.
    const FNO_EQUITIES: &[&str] = &[
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
        "BHARTIARTL", "BAJFINANCE", "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "AXISBANK",
        "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "BAJAJFINSV",
        "ADANIENT", "ADANIPORTS", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO", "ONGC",
        "NTPC", "POWERGRID", "COALINDIA", "TECHM", "INDUSINDBK", "DRREDDY", "CIPLA", "DIVISLAB",
        "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "BRITANNIA", "GRASIM", "SHREECEM", "TATACONSUM",
        "APOLLOHOSP", "HDFCLIFE", "SBILIFE", "M&M", "UPL", "BPCL", "DLF", "VEDL", "SAIL",
        "IOC", "GAIL", "TATAPOWER", "INDIGO", "HAVELLS", "DABUR", "GODREJCP", "BERGEPAINT",
        "AMBUJACEM", "BOSCHLTD", "BEL", "BHEL", "CUMMINSIND", "ASHOKLEY", "TVSMOTOR", "MRF",
        "LUPIN", "TORNTPHARM", "AUROPHARMA", "BIOCON", "GLENMARK", "ABCAPITAL", "CONCOR",
        "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "AUBANK", "YESBANK", "PERSISTENT", "COFORGE",
        "MPHASIS", "KPITTECH", "LTTS", "MARICO", "COLPAL", "GODREJPROP", "OBEROIRLTY",
        "AARTIIND", "ATUL", "DEEPAKNTR", "CYIENT", "NATIONALUM", "NMDC", "OIL", "PETRONET",
        "IGL", "MGL",
    ];
    FNO_EQUITIES.contains(&base)
}

fn leg(l: &Option<ChainLeg>) -> OptionLeg {
    match l {
        Some(v) => OptionLeg {
            oi: v.open_interest,
            oi_change: v.change_oi,
            volume: v.volume,
            ltp: v.last_price,
            iv: v.iv,
        },
        None => OptionLeg::default(),
    }
}

/// Assemble a chain, tagging the strike nearest the underlying as ATM.
///
/// Pure function so the shape is unit-testable without network access.
/// Crate-visible because the response DTO stays private; callers use
/// [`NseOptionsProvider::fetch_chain`].
pub(crate) fn build_chain(
    symbol: &str,
    expiry: Option<&str>,
    body: &ChainResponse,
) -> Result<OptionChain> {
    if body.records.data.is_empty() {
        return Err(BtError::EmptySeries(format!("no chain rows for {symbol}")));
    }
    let underlying = body.records.underlying_value.unwrap_or(0.0);
    let mut strikes: Vec<OptionStrike> = body
        .records
        .data
        .iter()
        .map(|r| OptionStrike {
            strike: r.strike_price,
            call: leg(&r.ce),
            put: leg(&r.pe),
            atm: false,
        })
        .collect();
    strikes.sort_by(|a, b| a.strike.partial_cmp(&b.strike).unwrap_or(std::cmp::Ordering::Equal));
    if underlying > 0.0 {
        if let Some(atm) = strikes
            .iter_mut()
            .min_by(|a, b| {
                (a.strike - underlying)
                    .abs()
                    .partial_cmp(&(b.strike - underlying).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            atm.atm = true;
        }
    }
    let resolved_expiry = expiry.map(str::to_string).unwrap_or_else(|| {
        body.records
            .expiry_dates
            .first()
            .cloned()
            .unwrap_or_default()
    });
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(OptionChain {
        symbol: symbol.to_uppercase(),
        expiry: resolved_expiry,
        underlying_value: underlying,
        strikes,
        fetched_at: now,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::json!({
            "records": {
                "expiryDates": ["30-Oct-2026", "26-Nov-2026"],
                "underlyingValue": 25750.5,
                "timestamp": "30-Oct-2026 15:30:00",
                "data": [
                    {"strikePrice": 25700.0, "expiryDate": "30-Oct-2026",
                     "CE": {"openInterest": 12345, "changeinOpenInterest": 2100,
                            "totalTradedVolume": 50000, "lastPrice": 145.3,
                            "impliedVolatility": 12.5},
                     "PE": {"openInterest": 8900, "changeinOpenInterest": -1200,
                            "totalTradedVolume": 30000, "lastPrice": 12.4,
                            "impliedVolatility": 13.1}},
                    {"strikePrice": 25800.0, "expiryDate": "30-Oct-2026",
                     "CE": null,
                     "PE": {"openInterest": 15400, "changeinOpenInterest": 3100,
                            "totalTradedVolume": 40000, "lastPrice": 25.6,
                            "impliedVolatility": 12.9}}
                ]
            },
            "filtered": {"data": []}
        })
    }

    #[test]
    fn parses_chain_tags_atm_and_tolerates_null_legs() {
        let body: ChainResponse = serde_json::from_value(fixture()).unwrap();
        let chain = build_chain("NIFTY", None, &body).unwrap();
        assert_eq!(chain.symbol, "NIFTY");
        assert_eq!(chain.expiry, "30-Oct-2026");
        assert_eq!(chain.underlying_value, 25750.5);
        assert_eq!(chain.strikes.len(), 2);
        // |25800 − 25750.5| = 49.5 beats |25700 − 25750.5| = 50.5.
        assert!(!chain.strikes[0].atm);
        assert!(chain.strikes[1].atm);
        assert_eq!(chain.strikes[0].call.oi, 12345);
        assert_eq!(chain.strikes[0].call.oi_change, 2100);
        assert!((chain.strikes[0].call.iv - 12.5).abs() < 1e-9);
        // Null CE becomes a zero leg, not a panic.
        assert_eq!(chain.strikes[1].call, OptionLeg::default());
        assert_eq!(chain.strikes[1].put.volume, 40000);
    }

    #[test]
    fn empty_data_is_an_error_not_an_empty_chain() {
        let body: ChainResponse = serde_json::from_value(serde_json::json!({
            "records": {"expiryDates": [], "data": []},
            "filtered": {"data": []}
        }))
        .unwrap();
        assert!(build_chain("NIFTY", None, &body).is_err());
    }

    #[test]
    fn fno_gate_keeps_indices_etfs_and_cash_names_out() {
        assert!(is_fno_symbol("NIFTY"));
        assert!(is_fno_symbol("BANKNIFTY"));
        assert!(is_fno_symbol("RELIANCE.NS"));
        assert!(is_fno_symbol("RELIANCE.BO"));
        assert!(is_fno_symbol("TCS"));
        assert!(!is_fno_symbol("GOLDBEES.NS"));
        assert!(!is_fno_symbol("^NSEI"));
        assert!(!is_fno_symbol("BTC-USD"));
        assert!(!is_fno_symbol("USDINR=X"));
    }

    #[test]
    fn chain_url_routes_indices_and_equities() {
        let idx = NseOptionsProvider::chain_url("NIFTY", None);
        assert!(idx.contains("option-chain-indices"));
        let eq = NseOptionsProvider::chain_url("RELIANCE", Some("30-Oct-2026"));
        assert!(eq.contains("option-chain-equities"));
        assert!(eq.contains("expiryDate="));
    }

    /// Live checks. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to nseindia.com"]
    async fn live_nifty_chain() {
        let p = NseOptionsProvider::new();
        p.init_session().await.unwrap();
        let chain = p.fetch_nifty_chain().await.unwrap();
        assert!(!chain.strikes.is_empty());
    }
}

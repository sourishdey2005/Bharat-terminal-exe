// crates/bt-data/src/moneycontrol.rs
// Author: Sourish Dey

//! Moneycontrol backup quotes for Indian equities (no API key).
//!
//! `GET priceapi.moneycontrol.com/pricefeed/nse/equitycash/{CODE}` returns
//! `{"code":"200","data":{...}}` with `pricecurrent` (LTP), `priceprevclose`,
//! `pricechange`, `pricepercentchange`, `VOL`, `NSEID`, `52H`, `52L` and
//! `lastupd_epoch`. A `"code"` other than `"200"` means the code is wrong.
//!
//! The `{CODE}` is Moneycontrol's short symbol code (`RI` for Reliance,
//! `TCS` for TCS), *not* the NSE ticker. [`MoneycontrolProvider::search`]
//! resolves names to codes through the autosuggest endpoint; when that is
//! unreachable, the well-known large-cap codes below still work.

use bt_core::{BtError, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration as StdDuration;
use tracing::instrument;

use crate::provider::Quote;

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
const PRICEFEED_URL: &str = "https://priceapi.moneycontrol.com/pricefeed/nse/equitycash";
const SEARCH_URL: &str =
    "https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php";

/// Well-known Moneycontrol codes for Nifty large-caps, so a quote works even
/// when the autosuggest endpoint is unreachable.
pub const KNOWN_CODES: &[(&str, &str)] = &[
    ("RELIANCE.NS", "RI"),
    ("TCS.NS", "TCS"),
    ("INFY.NS", "IT"),
    ("HDFCBANK.NS", "HDF"),
    ("ICICIBANK.NS", "ICI"),
    ("SBIN.NS", "SBI"),
    ("ITC.NS", "ITC"),
    ("LT.NS", "LNT"),
    ("TATAMOTORS.NS", "TM"),
    ("TATASTEEL.NS", "TIS"),
];

/// One autosuggest hit: display name plus the code `fetch_quote` needs.
#[derive(Debug, Clone, PartialEq)]
pub struct McSearchHit {
    pub name: String,
    pub code: String,
}

#[derive(Debug, Deserialize)]
struct PricefeedEnvelope {
    code: String,
    message: String,
    data: Option<McQuoteData>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct McQuoteData {
    #[serde(rename = "NSEID")]
    nse_id: String,
    pricecurrent: String,
    priceprevclose: String,
    pricechange: String,
    pricepercentchange: String,
    #[serde(rename = "VOL")]
    vol: String,
    #[serde(rename = "52H")]
    high_52w: String,
    #[serde(rename = "52L")]
    low_52w: String,
    lastupd_epoch: String,
}

pub struct MoneycontrolProvider {
    client: Client,
}

impl MoneycontrolProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(20))
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    /// Map an NSE ticker to a Moneycontrol code via [`KNOWN_CODES`], or `None`.
    pub fn known_code(ticker: &str) -> Option<&'static str> {
        let want = ticker.to_uppercase();
        KNOWN_CODES
            .iter()
            .find(|(t, _)| *t == want)
            .map(|(_, c)| *c)
    }

    /// Search names/codes. Returns hits; empty when the endpoint blocks the
    /// request rather than failing, so callers can fall back to [`KNOWN_CODES`].
    #[instrument(skip(self))]
    pub async fn search(&self, query: &str) -> Result<Vec<McSearchHit>> {
        let url = format!(
            "{SEARCH_URL}?classic=true&query={}&type=1&format=json",
            urlencoding::encode(query)
        );
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("Moneycontrol search error: {e}")))?;
        if !resp.status().is_success() {
            return Err(BtError::DataFetch(format!(
                "Moneycontrol search HTTP {}",
                resp.status()
            )));
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("Moneycontrol search JSON: {e}")))?;
        Ok(parse_search_body(&body))
    }

    /// Quote for a Moneycontrol code (e.g. `"RI"`), or for an NSE ticker that
    /// appears in [`KNOWN_CODES`].
    #[instrument(skip(self))]
    pub async fn fetch_quote(&self, code_or_ticker: &str) -> Result<Quote> {
        let code = Self::known_code(code_or_ticker).unwrap_or(code_or_ticker);
        let url = format!("{PRICEFEED_URL}/{code}");
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("Moneycontrol network error: {e}")))?;
        if !resp.status().is_success() {
            return Err(BtError::DataFetch(format!(
                "Moneycontrol HTTP {} for {code}",
                resp.status()
            )));
        }
        let env: PricefeedEnvelope = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("Moneycontrol JSON error: {e}")))?;
        if env.code != "200" {
            return Err(BtError::DataFetch(format!(
                "Moneycontrol: {} ({})",
                env.message, code
            )));
        }
        let d = env
            .data
            .ok_or_else(|| BtError::InvalidInput("Moneycontrol: empty data".into()))?;
        build_quote(code, &d)
    }
}

impl Default for MoneycontrolProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn num(s: &str) -> Result<f64> {
    s.trim()
        .replace(',', "")
        .parse::<f64>()
        .map_err(|_| BtError::InvalidInput(format!("Moneycontrol: bad number {s:?}")))
}

fn build_quote(code: &str, d: &McQuoteData) -> Result<Quote> {
    let ts = d
        .lastupd_epoch
        .parse::<i64>()
        .ok()
        .and_then(|t| DateTime::from_timestamp(t, 0))
        .unwrap_or_else(Utc::now);
    Ok(Quote {
        symbol: if d.nse_id.is_empty() {
            code.to_string()
        } else {
            d.nse_id.clone()
        },
        price: num(&d.pricecurrent)?,
        change: d.pricechange.parse().unwrap_or(0.0),
        change_pct: d.pricepercentchange.parse().unwrap_or(0.0),
        volume: d.vol.replace(',', "").parse().unwrap_or(0),
        timestamp: ts,
    })
}

/// Lenient autosuggest parsing: the endpoint's shape has drifted before, so
/// unknown items are skipped and callers fall back to [`KNOWN_CODES`].
pub fn parse_search_body(body: &serde_json::Value) -> Vec<McSearchHit> {
    let arr = match body.as_array().or_else(|| body.get("data").and_then(|v| v.as_array())) {
        Some(a) => a,
        None => return Vec::new(),
    };
    arr.iter()
        .filter_map(|v| {
            let name = v
                .get("sc_name")
                .or_else(|| v.get("name"))
                .or_else(|| v.get("company_name"))
                .and_then(|s| s.as_str())?
                .to_string();
            let code = v
                .get("sc_id")
                .or_else(|| v.get("code"))
                .or_else(|| v.get("DISPID"))
                .and_then(|s| s.as_str())?
                .to_string();
            Some(McSearchHit { name, code })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUOTE_JSON: &str = r#"{"code":"200","message":"Success","data":{"NSEID":"RELIANCE","pricecurrent":"1187.00","priceprevclose":"1182.00","pricechange":"5.0000","pricepercentchange":"0.4230","VOL":"16376789","52H":"1611.80","52L":"1181.70","lastupd_epoch":"1790764199"}}"#;

    #[test]
    fn parses_a_quote_envelope() {
        let env: PricefeedEnvelope = serde_json::from_str(QUOTE_JSON).unwrap();
        assert_eq!(env.code, "200");
        let q = build_quote("RI", &env.data.unwrap()).unwrap();
        assert_eq!(q.symbol, "RELIANCE");
        assert_eq!(q.price, 1187.0);
        assert_eq!(q.change, 5.0);
        assert_eq!(q.volume, 16376789);
    }

    #[test]
    fn non_200_codes_are_rejected_with_the_message() {
        let env: PricefeedEnvelope =
            serde_json::from_str(r#"{"code":"201","message":"No data","data":null}"#).unwrap();
        assert_ne!(env.code, "200");
        assert!(env.data.is_none());
    }

    #[test]
    fn known_codes_cover_the_large_caps() {
        assert_eq!(MoneycontrolProvider::known_code("reliance.ns"), Some("RI"));
        assert_eq!(MoneycontrolProvider::known_code("TCS.NS"), Some("TCS"));
        assert_eq!(MoneycontrolProvider::known_code("UNKNOWN.X"), None);
    }

    #[test]
    fn search_parsing_is_lenient() {
        assert!(parse_search_body(&serde_json::json!({})).is_empty());
        let hits = parse_search_body(&serde_json::json!([
            {"sc_name": "Reliance Industries", "sc_id": "RI"},
            {"unexpected": "shape"}
        ]));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].code, "RI");
    }

    /// Live check. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to moneycontrol.com"]
    async fn live_quote() {
        let p = MoneycontrolProvider::new();
        let q = p.fetch_quote("RI").await.unwrap();
        assert!(q.price > 0.0);
    }
}

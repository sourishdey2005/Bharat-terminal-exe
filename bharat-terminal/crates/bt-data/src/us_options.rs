// crates/bt-data/src/us_options.rs
// Author: Sourish Dey

//! US option chains over Yahoo Finance (no API key).
//!
//! `GET query2.../v7/finance/options/{SYM}[?date={unix}]` returns
//! `optionChain.result[0]` with `expirationDates`, `strikes`, the underlying
//! quote and one `options[]` entry holding `calls`/`puts`. Each leg carries
//! strike, openInterest, volume, lastPrice, impliedVolatility, bid and ask;
//! Greeks are computed locally in bt-analytics (`options` module), never
//! trusted from a feed.
//!
//! Yahoo gates this endpoint behind a cookie + crumb handshake: warm up on the
//! homepage, fetch a crumb from `v1/test/getcrumb`, then call the API with
//! `?crumb=`. A 401 means the crumb went stale, so it is refreshed once and the
//! call retried before giving up. Everything else fails closed.

use bt_core::{BtError, OptionChain, OptionLeg, OptionStrike, Result};
use reqwest::cookie::Jar;
use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";
const HOME_URL: &str = "https://finance.yahoo.com";
const CRUMB_URL: &str = "https://query2.finance.yahoo.com/v1/test/getcrumb";
const OPTIONS_URL: &str = "https://query2.finance.yahoo.com/v7/finance/options";

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct YahooLeg {
    #[serde(rename = "contractSymbol", default)]
    contract_symbol: String,
    #[serde(default)]
    strike: f64,
    #[serde(rename = "lastPrice", default)]
    last_price: f64,
    #[serde(default)]
    bid: f64,
    #[serde(rename = "ask", default)]
    ask: f64,
    #[serde(default)]
    volume: u64,
    #[serde(rename = "openInterest", default)]
    open_interest: u64,
    #[serde(rename = "impliedVolatility", default)]
    iv: f64,
    #[serde(rename = "inTheMoney", default)]
    itm: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct YahooOptionsNode {
    #[serde(rename = "expirationDate", default)]
    expiration_date: i64,
    #[serde(default)]
    calls: Vec<YahooLeg>,
    #[serde(default)]
    puts: Vec<YahooLeg>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct YahooQuote {
    #[serde(rename = "regularMarketPrice", default)]
    price: f64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct YahooResult {
    #[serde(rename = "underlyingSymbol", default)]
    symbol: String,
    #[serde(rename = "expirationDates", default)]
    expirations: Vec<i64>,
    #[serde(default)]
    strikes: Vec<f64>,
    #[serde(default)]
    options: Vec<YahooOptionsNode>,
    #[serde(default)]
    quote: Option<YahooQuote>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct YahooChain {
    #[serde(default)]
    result: Vec<YahooResult>,
    error: Option<serde_json::Value>,
}

pub struct UsOptionsProvider {
    client: Client,
}

impl UsOptionsProvider {
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

    /// Seed cookies from the homepage. Best-effort: a failure here does not
    /// fail the call, because the crumb step may still succeed.
    async fn warmup(&self) {
        let _ = self.client.get(HOME_URL).send().await;
    }

    /// Fetch a fresh crumb with the current cookie jar.
    async fn crumb(&self) -> Result<String> {
        let text = self
            .client
            .get(CRUMB_URL)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("Yahoo crumb network error: {e}")))?
            .error_for_status()
            .map_err(|e| BtError::DataFetch(format!("Yahoo crumb HTTP error: {e}")))?
            .text()
            .await
            .map_err(|e| BtError::DataFetch(format!("Yahoo crumb body error: {e}")))?;
        if text.is_empty() || text.len() > 64 {
            return Err(BtError::DataFetch("Yahoo returned an unusable crumb".into()));
        }
        Ok(text)
    }

    async fn get_chain(&self, url: &str, crumb: &str) -> Result<YahooChain> {
        let sep = if url.contains('?') { '&' } else { '?' };
        let full = format!("{url}{sep}crumb={crumb}");
        let resp = self
            .client
            .get(&full)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("Yahoo options network error: {e}")))?;
        if resp.status() == 401 {
            return Err(BtError::DataFetch("Yahoo 401: crumb stale".into()));
        }
        if !resp.status().is_success() {
            return Err(BtError::DataFetch(format!(
                "Yahoo options HTTP {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("Yahoo options JSON error: {e}")))
    }

    /// Listed expiry timestamps (unix seconds) for a US symbol.
    #[instrument(skip(self))]
    pub async fn fetch_us_expiries(&self, symbol: &str) -> Result<Vec<i64>> {
        let chain = self.fetch_chain_raw(symbol, None).await?;
        Ok(chain.expirations)
    }

    /// Full chain for a US symbol, defaulting to the nearest expiry.
    /// `expiry` is a unix timestamp as listed by [`Self::fetch_us_expiries`].
    #[instrument(skip(self))]
    pub async fn fetch_us_chain(
        &self,
        symbol: &str,
        expiry: Option<i64>,
    ) -> Result<OptionChain> {
        let chain = self.fetch_chain_raw(symbol, expiry).await?;
        build_us_chain(symbol, expiry, &chain)
    }

    async fn fetch_chain_raw(&self, symbol: &str, expiry: Option<i64>) -> Result<YahooResult> {
        let upper = symbol.to_uppercase();
        let mut url = format!("{OPTIONS_URL}/{upper}");
        if let Some(e) = expiry {
            url.push_str(&format!("?date={e}"));
        }
        self.warmup().await;
        let mut crumb = self.crumb().await?;
        let mut body = self.get_chain(&url, &crumb).await;
        // One stale-crumb retry: refresh and replay exactly once.
        if matches!(&body, Err(BtError::DataFetch(m)) if m.contains("401")) {
            crumb = self.crumb().await?;
            body = self.get_chain(&url, &crumb).await;
        }
        let body = body?;
        body.result.into_iter().next().ok_or_else(|| {
            BtError::EmptySeries(format!("no Yahoo options result for {upper}"))
        })
    }
}

impl Default for UsOptionsProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn leg(l: &YahooLeg) -> OptionLeg {
    OptionLeg {
        oi: l.open_interest,
        oi_change: 0,
        volume: l.volume,
        ltp: l.last_price,
        iv: l.iv,
    }
}

/// Merge one Yahoo `options[]` node into the shared chain shape.
///
/// Calls and puts are joined on strike; a strike present on only one side gets
/// a zero leg rather than being dropped, so the row count always matches the
/// exchange's strike list. Crate-visible because the Yahoo DTO stays private;
/// callers use [`UsOptionsProvider::fetch_us_chain`]. Pure function,
/// fixture-tested.
pub(crate) fn build_us_chain(
    symbol: &str,
    requested_expiry: Option<i64>,
    result: &YahooResult,
) -> Result<OptionChain> {
    let node = result.options.first().ok_or_else(|| {
        BtError::EmptySeries(format!("no Yahoo options node for {symbol}"))
    })?;
    use std::collections::BTreeMap;
    let mut rows: BTreeMap<u64, OptionStrike> = BTreeMap::new();
    for c in &node.calls {
        let key = c.strike.to_bits();
        let row = rows.entry(key).or_insert_with(|| OptionStrike {
            strike: c.strike,
            ..Default::default()
        });
        row.call = leg(c);
    }
    for p in &node.puts {
        let key = p.strike.to_bits();
        let row = rows.entry(key).or_insert_with(|| OptionStrike {
            strike: p.strike,
            ..Default::default()
        });
        row.put = leg(p);
    }
    if rows.is_empty() {
        return Err(BtError::EmptySeries(format!(
            "empty Yahoo options node for {symbol}"
        )));
    }
    let mut strikes: Vec<OptionStrike> = rows.into_values().collect();
    let underlying = result.quote.as_ref().map(|q| q.price).unwrap_or(0.0);
    if underlying > 0.0 {
        if let Some(atm) = strikes.iter_mut().min_by(|a, b| {
            (a.strike - underlying)
                .abs()
                .partial_cmp(&(b.strike - underlying).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            atm.atm = true;
        }
    }
    let expiry = requested_expiry
        .map(|e| e.to_string())
        .unwrap_or_else(|| node.expiration_date.to_string());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(OptionChain {
        symbol: symbol.to_uppercase(),
        expiry,
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
            "underlyingSymbol": "AAPL",
            "expirationDates": [1790908800, 1791513600],
            "strikes": [270.0, 275.0],
            "quote": {"regularMarketPrice": 273.5},
            "options": [{
                "expirationDate": 1790908800,
                "calls": [
                    {"contractSymbol": "AAPL1", "strike": 270.0, "lastPrice": 5.2,
                     "bid": 5.0, "ask": 5.4, "volume": 1200,
                     "openInterest": 3400, "impliedVolatility": 0.22, "inTheMoney": true},
                    {"contractSymbol": "AAPL2", "strike": 275.0, "lastPrice": 2.1,
                     "bid": 2.0, "ask": 2.2, "volume": 800,
                     "openInterest": 2100, "impliedVolatility": 0.21, "inTheMoney": false}
                ],
                "puts": [
                    {"contractSymbol": "AAPL3", "strike": 270.0, "lastPrice": 1.8,
                     "bid": 1.7, "ask": 1.9, "volume": 900,
                     "openInterest": 2800, "impliedVolatility": 0.23, "inTheMoney": false}
                ]
            }]
        })
    }

    #[test]
    fn merges_calls_and_puts_on_strike() {
        let result: YahooResult = serde_json::from_value(fixture()).unwrap();
        let chain = build_us_chain("AAPL", None, &result).unwrap();
        assert_eq!(chain.symbol, "AAPL");
        assert_eq!(chain.expiry, "1790908800");
        assert_eq!(chain.strikes.len(), 2);
        assert_eq!(chain.underlying_value, 273.5);
        // 275.0 is nearer 273.5 than 270.0 is.
        assert!(!chain.strikes[0].atm);
        assert!(chain.strikes[1].atm);
        assert_eq!(chain.strikes[0].call.oi, 3400);
        assert_eq!(chain.strikes[0].put.volume, 900);
        // 275 put absent on the exchange: zero leg, row kept.
        assert_eq!(chain.strikes[1].put, OptionLeg::default());
        assert!((chain.strikes[1].call.iv - 0.21).abs() < 1e-9);
    }

    #[test]
    fn empty_nodes_are_an_error() {
        let mut result: YahooResult = serde_json::from_value(fixture()).unwrap();
        result.options.clear();
        assert!(build_us_chain("AAPL", None, &result).is_err());
    }

    /// Live check. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to finance.yahoo.com"]
    async fn live_aapl_chain() {
        let p = UsOptionsProvider::new();
        let chain = p.fetch_us_chain("AAPL", None).await.unwrap();
        assert!(!chain.strikes.is_empty());
    }
}

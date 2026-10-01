// crates/bt-data/src/binance.rs
// Author: Sourish Dey

//! Binance public market data (no API key).
//!
//! Backup crypto source alongside Coinbase: `GET /api/v3/klines` returns
//! `[open_time_ms, open, high, low, close, volume, ...]` arrays, oldest or
//! newest depending on parameters (default oldest-first for a bounded range).

use bt_core::{BtError, Candle, OhlcvSeries, Result};
use reqwest::Client;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (compatible; BharatTerminal/4.0)";
const BASE_URL: &str = "https://api.binance.com";

/// Binance intervals accepted by `/api/v3/klines`.
#[derive(Debug, Clone, Copy)]
pub enum BinanceInterval {
    Min1,
    Min5,
    Min15,
    Hour1,
    Hour4,
    Day1,
    Week1,
}

impl BinanceInterval {
    fn as_str(self) -> &'static str {
        match self {
            BinanceInterval::Min1 => "1m",
            BinanceInterval::Min5 => "5m",
            BinanceInterval::Min15 => "15m",
            BinanceInterval::Hour1 => "1h",
            BinanceInterval::Hour4 => "4h",
            BinanceInterval::Day1 => "1d",
            BinanceInterval::Week1 => "1w",
        }
    }
}

pub struct BinanceProvider {
    client: Client,
}

impl BinanceProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(20))
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    /// Fetch klines for e.g. `BTCUSDT` (no separator, quote asset appended).
    ///
    /// Returns oldest-first candles. `limit` is clamped to 1..=1000 by the API.
    #[instrument(skip(self))]
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: BinanceInterval,
        limit: u32,
    ) -> Result<OhlcvSeries> {
        let limit = limit.clamp(1, 1000);
        let url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&limit={}",
            BASE_URL,
            symbol.to_uppercase().replace(['-', '/'], ""),
            interval.as_str(),
            limit
        );
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("Binance network error: {e}")))?;
        if !resp.status().is_success() {
            return Err(BtError::DataFetch(format!(
                "Binance HTTP {} for {symbol}",
                resp.status()
            )));
        }
        let rows: Vec<Vec<serde_json::Value>> = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("Binance JSON error: {e}")))?;
        let candles = parse_klines(&rows)?;
        if candles.is_empty() {
            return Err(BtError::EmptySeries(symbol.to_string()));
        }
        Ok(OhlcvSeries::new(symbol, candles))
    }
}

impl Default for BinanceProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse raw kline rows into candles. Pure function so it is unit-testable
/// without network access.
pub fn parse_klines(rows: &[Vec<serde_json::Value>]) -> Result<Vec<Candle>> {
    let mut out = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let num = |idx: usize| -> Result<f64> {
            r.get(idx)
                .and_then(|v| {
                    v.as_f64().or_else(|| {
                        v.as_str()
                            .and_then(|s| s.parse::<f64>().ok())
                            .or_else(|| v.as_u64().map(|n| n as f64))
                    })
                })
                .ok_or_else(|| BtError::InvalidInput(format!("bad kline field {idx} in row {i}")))
        };
        // Binance times are milliseconds; the rest of the app works in seconds.
        out.push(Candle::new(
            num(0)? / 1000.0,
            num(1)?,
            num(2)?,
            num(3)?,
            num(4)?,
            num(5)?,
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(ms: u64, o: &str, h: &str, l: &str, c: &str, v: &str) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!(ms),
            serde_json::json!(o),
            serde_json::json!(h),
            serde_json::json!(l),
            serde_json::json!(c),
            serde_json::json!(v),
        ]
    }

    #[test]
    fn parses_kline_rows_with_string_numbers() {
        let rows = vec![
            row(1790640000000, "83500.01", "84563.99", "82775.94", "83663.66", "1234.5"),
            row(1790726400000, "83663.66", "84000.00", "83000.00", "83800.00", "900.0"),
        ];
        let cs = parse_klines(&rows).unwrap();
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0].t, 1790640000.0);
        assert_eq!(cs[0].open, 83500.01);
        assert_eq!(cs[0].high, 84563.99);
        assert_eq!(cs[0].low, 82775.94);
        assert_eq!(cs[0].close, 83663.66);
        assert_eq!(cs[1].t, 1790726400.0);
    }

    #[test]
    fn short_rows_are_rejected_not_panicked() {
        let rows = vec![vec![serde_json::json!(1)]];
        assert!(parse_klines(&rows).is_err());
        assert!(parse_klines(&[]).unwrap().is_empty());
    }

    /// Live check. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to api.binance.com"]
    async fn live_klines() {
        let p = BinanceProvider::new();
        let s = p
            .fetch_klines("BTCUSDT", BinanceInterval::Day1, 5)
            .await
            .unwrap();
        assert!(!s.candles.is_empty());
    }
}

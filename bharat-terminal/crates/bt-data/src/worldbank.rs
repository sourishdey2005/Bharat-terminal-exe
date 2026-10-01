// crates/bt-data/src/worldbank.rs
// Author: Sourish Dey

//! World Bank macro indicators (no API key).
//!
//! `GET /v2/country/{cc}/indicator/{code}?format=json` returns a two-element
//! array: `[page_metadata, [ {date, value, ...}, ... ]]` with newest first.
//! Values arrive as numbers or `null` for missing years.

use bt_core::{BtError, Result};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (compatible; BharatTerminal/4.0)";
const BASE_URL: &str = "https://api.worldbank.org/v2";

/// Well-known indicator codes used by the Macro views.
pub mod code {
    /// GDP in current US$.
    pub const GDP_USD: &str = "NY.GDP.MKTP.CD";
    /// Inflation, consumer prices (annual %).
    pub const INFLATION_CPI: &str = "FP.CPI.TOTL.ZG";
    /// Unemployment, total (% of labor force).
    pub const UNEMPLOYMENT: &str = "SL.UEM.TOTL.NE.ZS";
    /// Total population.
    pub const POPULATION: &str = "SP.POP.TOTL";
    /// Central government debt, total (% of GDP).
    pub const DEBT_PCT_GDP: &str = "GC.DOD.TOTL.GD.ZS";
    /// Real interest rate (%).
    pub const REAL_RATE: &str = "FR.INR.RINR";
}

/// One annual observation. `value` is `None` for years the Bank has no data.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroPoint {
    pub year: i32,
    pub value: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WbRow {
    date: String,
    value: Option<f64>,
}

pub struct WorldBankProvider {
    client: Client,
}

impl WorldBankProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(20))
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    /// Fetch an indicator for an ISO-2 country code (e.g. `"IN"`, `"US"`),
    /// oldest-first. `per_page` caps the rows (the API pages at 1000 by default).
    #[instrument(skip(self))]
    pub async fn fetch_indicator(
        &self,
        country: &str,
        indicator: &str,
        per_page: u32,
    ) -> Result<Vec<MacroPoint>> {
        let url = format!(
            "{BASE_URL}/country/{}/indicator/{}?format=json&per_page={}",
            country.to_uppercase(),
            indicator,
            per_page.clamp(1, 20000)
        );
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("World Bank network error: {e}")))?;
        if !resp.status().is_success() {
            return Err(BtError::DataFetch(format!(
                "World Bank HTTP {} for {country}/{indicator}",
                resp.status()
            )));
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("World Bank JSON error: {e}")))?;
        let mut pts = parse_indicator_body(&body)?;
        // API returns newest-first; callers want chronological order.
        pts.reverse();
        Ok(pts)
    }

    /// GDP in current US$ for a country.
    pub async fn fetch_gdp(&self, country: &str) -> Result<Vec<MacroPoint>> {
        self.fetch_indicator(country, code::GDP_USD, 70).await
    }

    /// CPI inflation (annual %) for a country.
    pub async fn fetch_inflation(&self, country: &str) -> Result<Vec<MacroPoint>> {
        self.fetch_indicator(country, code::INFLATION_CPI, 70).await
    }

    /// Unemployment (% of labor force) for a country.
    pub async fn fetch_unemployment(&self, country: &str) -> Result<Vec<MacroPoint>> {
        self.fetch_indicator(country, code::UNEMPLOYMENT, 70).await
    }

    /// Total population for a country.
    pub async fn fetch_population(&self, country: &str) -> Result<Vec<MacroPoint>> {
        self.fetch_indicator(country, code::POPULATION, 70).await
    }
}

impl Default for WorldBankProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse the `[meta, rows]` envelope. Pure function, tested with fixtures.
pub fn parse_indicator_body(body: &serde_json::Value) -> Result<Vec<MacroPoint>> {
    let rows = body
        .get(1)
        .and_then(|v| v.as_array())
        .ok_or_else(|| BtError::InvalidInput("World Bank: expected [meta, rows]".into()))?;
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        // Rows occasionally carry non-object sentinels; skip rather than fail.
        let row: WbRow = match serde_json::from_value(r.clone()) {
            Ok(row) => row,
            Err(_) => continue,
        };
        let year: i32 = row.date.parse().map_err(|_| {
            BtError::InvalidInput(format!("World Bank: bad year {:?}", row.date))
        })?;
        out.push(MacroPoint {
            year,
            value: row.value,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::json!([
            {"page":1,"pages":22,"per_page":3,"total":66},
            [
                {"date":"2024","value":3910.0},
                {"date":"2023","value":null},
                {"date":"2022","value":3549.0}
            ]
        ])
    }

    #[test]
    fn parses_rows_and_keeps_nulls() {
        let pts = parse_indicator_body(&fixture()).unwrap();
        assert_eq!(pts.len(), 3);
        assert_eq!(pts[0].year, 2024);
        assert_eq!(pts[0].value, Some(3910.0));
        assert_eq!(pts[1].value, None);
    }

    #[test]
    fn malformed_envelopes_are_rejected() {
        assert!(parse_indicator_body(&serde_json::json!({})).is_err());
        assert!(parse_indicator_body(&serde_json::json!([{"page":1}])).is_err());
    }

    /// Live check. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to api.worldbank.org"]
    async fn live_gdp_india() {
        let p = WorldBankProvider::new();
        let pts = p.fetch_gdp("IN").await.unwrap();
        assert!(pts.iter().any(|pt| pt.value.unwrap_or(0.0) > 0.0));
    }
}

// crates/bt-data/src/amfi.rs
// Author: Sourish Dey

//! AMFI mutual-fund NAVs (no API key).
//!
//! `GET https://www.amfiindia.com/spages/NAVAll.txt` returns the day's NAVs as
//! semicolon-separated text. Fund-house headers and section titles do not start
//! with a scheme code, so data rows are exactly the lines whose first field is
//! numeric: `code;name;isin_a;isin_b;nav;repurchase;sale;date`.

use bt_core::{BtError, Result};
use reqwest::Client;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (compatible; BharatTerminal/4.0)";
const NAV_URL: &str = "https://www.amfiindia.com/spages/NAVAll.txt";

/// One scheme's published NAV.
#[derive(Debug, Clone, PartialEq)]
pub struct NavEntry {
    pub scheme_code: String,
    pub scheme_name: String,
    pub nav: f64,
    /// As published, e.g. `30-Sep-2026`.
    pub date: String,
}

pub struct AmfiProvider {
    client: Client,
}

impl AmfiProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    /// All schemes published today (typically 10,000+ rows).
    #[instrument(skip(self))]
    pub async fn fetch_all_navs(&self) -> Result<Vec<NavEntry>> {
        let text = self
            .client
            .get(NAV_URL)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("AMFI network error: {e}")))?
            .error_for_status()
            .map_err(|e| BtError::DataFetch(format!("AMFI HTTP error: {e}")))?
            .text()
            .await
            .map_err(|e| BtError::DataFetch(format!("AMFI body error: {e}")))?;
        Ok(parse_nav_text(&text))
    }

    /// NAVs whose scheme name contains `query` (case-insensitive).
    #[instrument(skip(self))]
    pub async fn fetch_nav(&self, query: &str) -> Result<Vec<NavEntry>> {
        let all = self.fetch_all_navs().await?;
        let q = query.to_lowercase();
        Ok(all
            .into_iter()
            .filter(|e| e.scheme_name.to_lowercase().contains(&q))
            .collect())
    }
}

impl Default for AmfiProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse NAVAll.txt. Pure function, tested with fixtures.
pub fn parse_nav_text(text: &str) -> Vec<NavEntry> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let mut f = line.split(';');
        let code = f.next().unwrap_or("");
        // Data rows start with a numeric scheme code; everything else
        // (fund-house names, "Open Ended Schemes", column headers) is skipped.
        if code.is_empty() || !code.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let name = f.next().unwrap_or("").trim().to_string();
        let _isin_a = f.next();
        let _isin_b = f.next();
        let nav: f64 = match f.next().unwrap_or("").trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let _repurchase = f.next();
        let _sale = f.next();
        let date = f.next().unwrap_or("").trim().to_string();
        if name.is_empty() {
            continue;
        }
        out.push(NavEntry {
            scheme_code: code.to_string(),
            scheme_name: name,
            nav,
            date,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "Aditya Birla Sun Life Mutual Fund\n\
        Open Ended Schemes - Equity\n\
        Scheme Code;Scheme Name;ISIN Div Payout/ISIN Growth;ISIN Div Reinvestment;Net Asset Value;Repurchase Price;Sale Price;Date\n\
        119551;Aditya Birla Sun Life Banking & Financial Services Fund - Growth;INF209K01YM1;INF209K01YP5;36.12;35.76;36.12;30-Sep-2026\n\
        100028;Aditya Birla Sun Life Liquid Fund - Growth;INF209K01059;INF209K01AP8;500.25;500.25;500.25;30-Sep-2026\n\
        12;PRU ICICI GROWTH PLAN - GROWTH;INF109K01BL3;-;11.54;11.31;11.54;29-Sep-2026\n\
        Close Ended Schemes\n";

    #[test]
    fn parses_data_rows_and_skips_headers() {
        let out = parse_nav_text(FIXTURE);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].scheme_code, "119551");
        assert_eq!(out[0].nav, 36.12);
        assert_eq!(out[0].date, "30-Sep-2026");
        assert_eq!(out[2].scheme_code, "12");
    }

    #[test]
    fn bad_nav_values_are_skipped_not_fatal() {
        let out = parse_nav_text("12345;Some Fund;a;b;N/A;c;d;01-Jan-2026\n");
        assert!(out.is_empty());
    }

    /// Live check. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to amfiindia.com"]
    async fn live_navs() {
        let p = AmfiProvider::new();
        let all = p.fetch_all_navs().await.unwrap();
        assert!(all.len() > 1000);
    }
}

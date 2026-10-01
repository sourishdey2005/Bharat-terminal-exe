// crates/bt-data/src/sec.rs
// Author: Sourish Dey

//! SEC EDGAR company data (no API key).
//!
//! Three endpoints, all JSON:
//! - `www.sec.gov/files/company_tickers.json` maps tickers to CIKs.
//! - `data.sec.gov/submissions/CIK{cik:010}.json` lists recent filings with
//!   forms, dates and accession numbers (10-K, 10-Q, 8-K, 4, ...).
//! - `data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json` gives
//!   tagged XBRL facts (revenues, net income, assets).
//!
//! SEC asks callers to send a descriptive User-Agent; the one below identifies
//! the app. Insider trades are surfaced as Form 4 filing links with dates - the
//! submissions feed does not carry transaction detail, and parsing every Form 4
//! XML on demand would turn one request into dozens.

use bt_core::{BtError, Result};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "BharatTerminal/4.0 (Sourish Dey)";
const TICKERS_URL: &str = "https://www.sec.gov/files/company_tickers.json";

/// One filing from the submissions feed.
#[derive(Debug, Clone, PartialEq)]
pub struct SecFiling {
    pub form: String,
    pub filing_date: String,
    pub accession: String,
    pub primary_doc: String,
    /// Direct link to the filing index on sec.gov.
    pub url: String,
}

/// One insider (Form 4) event, as far as the submissions feed describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct InsiderTrade {
    pub filing_date: String,
    pub accession: String,
    pub url: String,
}

/// One annual XBRL fact (10-K only, to keep quarterly restatements out).
#[derive(Debug, Clone, PartialEq)]
pub struct XbrlPoint {
    /// Fiscal period end, `YYYY-MM-DD`.
    pub end: String,
    pub value: f64,
}

#[derive(Debug, Deserialize)]
struct TickerRow {
    cik_str: u32,
    ticker: String,
}

#[derive(Debug, Deserialize)]
struct Submissions {
    filings: SubmissionsFilings,
}

#[derive(Debug, Deserialize)]
struct SubmissionsFilings {
    recent: SubmissionsRecent,
}

#[derive(Debug, Deserialize)]
struct SubmissionsRecent {
    #[serde(rename = "accessionNumber")]
    accession: Vec<String>,
    #[serde(rename = "filingDate")]
    filing_date: Vec<String>,
    form: Vec<String>,
    #[serde(rename = "primaryDocument")]
    primary_doc: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ConceptUnits {
    #[serde(rename = "USD")]
    usd: Option<Vec<ConceptFact>>,
}

#[derive(Debug, Deserialize)]
struct ConceptBody {
    units: ConceptUnits,
}

#[derive(Debug, Deserialize)]
struct ConceptFact {
    val: f64,
    end: String,
    form: String,
}

pub struct SecProvider {
    client: Client,
}

impl SecProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(20))
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        self.client.get(url).header("Accept", "application/json")
    }

    /// Look up the zero-padded 10-digit CIK for a ticker (e.g. `"AAPL"`).
    #[instrument(skip(self))]
    pub async fn cik_for_ticker(&self, ticker: &str) -> Result<String> {
        let body: serde_json::Value = self
            .get(TICKERS_URL)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("SEC tickers network error: {e}")))?
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("SEC tickers JSON error: {e}")))?;
        let want = ticker.to_uppercase();
        let map = body.as_object().ok_or_else(|| {
            BtError::InvalidInput("SEC tickers: expected an object".into())
        })?;
        for v in map.values() {
            let row: TickerRow = serde_json::from_value(v.clone()).map_err(|e| {
                BtError::InvalidInput(format!("SEC tickers row error: {e}"))
            })?;
            if row.ticker.to_uppercase() == want {
                return Ok(format!("{:010}", row.cik_str));
            }
        }
        Err(BtError::InvalidInput(format!(
            "SEC: no CIK for ticker {ticker}"
        )))
    }

    /// Recent filings for a 10-digit CIK (or anything `cik_for_ticker` returns).
    #[instrument(skip(self))]
    pub async fn fetch_sec_filings(&self, cik: &str) -> Result<Vec<SecFiling>> {
        let url = format!("https://data.sec.gov/submissions/CIK{cik}.json");
        let body: Submissions = self
            .get(&url)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("SEC submissions network error: {e}")))?
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("SEC submissions JSON error: {e}")))?;
        Ok(build_filings(cik, &body.filings.recent))
    }

    /// Form 4 (insider) events for a CIK, newest first as filed.
    #[instrument(skip(self))]
    pub async fn fetch_insider_activity(&self, cik: &str) -> Result<Vec<InsiderTrade>> {
        let filings = self.fetch_sec_filings(cik).await?;
        Ok(filings
            .into_iter()
            .filter(|f| f.form == "4")
            .map(|f| InsiderTrade {
                filing_date: f.filing_date,
                accession: f.accession,
                url: f.url,
            })
            .collect())
    }

    /// Annual values of a us-gaap tag (e.g. `"Revenues"`, `"NetIncomeLoss"`,
    /// `"Assets"`) for a CIK, oldest-first.
    #[instrument(skip(self))]
    pub async fn fetch_xbrl_financials(&self, cik: &str, tag: &str) -> Result<Vec<XbrlPoint>> {
        let url = format!("https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json");
        let body: ConceptBody = self
            .get(&url)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("SEC XBRL network error: {e}")))?
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("SEC XBRL JSON error: {e}")))?;
        let mut pts: Vec<XbrlPoint> = body
            .units
            .usd
            .unwrap_or_default()
            .into_iter()
            .filter(|f| f.form == "10-K")
            .map(|f| XbrlPoint {
                end: f.end,
                value: f.val,
            })
            .collect();
        pts.sort_by(|a, b| a.end.cmp(&b.end));
        Ok(pts)
    }
}

impl Default for SecProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn archive_url(cik: &str, accession: &str, doc: &str) -> String {
    format!(
        "https://www.sec.gov/Archives/edgar/data/{}/{}/{doc}",
        cik.trim_start_matches('0'),
        accession.replace('-', "")
    )
}

fn build_filings(cik: &str, recent: &SubmissionsRecent) -> Vec<SecFiling> {
    let n = recent
        .form
        .len()
        .min(recent.accession.len())
        .min(recent.filing_date.len())
        .min(recent.primary_doc.len());
    (0..n)
        .map(|i| SecFiling {
            form: recent.form[i].clone(),
            filing_date: recent.filing_date[i].clone(),
            accession: recent.accession[i].clone(),
            primary_doc: recent.primary_doc[i].clone(),
            url: archive_url(cik, &recent.accession[i], &recent.primary_doc[i]),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recent_fixture() -> serde_json::Value {
        serde_json::json!({
            "accessionNumber": ["0000320193-26-000070", "0000320193-26-000066"],
            "filingDate": ["2026-09-25", "2026-09-18"],
            "form": ["4", "8-K"],
            "primaryDocument": ["wf-form4_123.xml", "aapl-20260918_8k.htm"]
        })
    }

    #[test]
    fn builds_filings_with_archive_links() {
        let recent: SubmissionsRecent = serde_json::from_value(recent_fixture()).unwrap();
        let out = build_filings("0000320193", &recent);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].form, "4");
        assert!(out[0].url.contains("Archives/edgar/data/320193/000032019326000070"));
        assert!(out[0].url.ends_with("wf-form4_123.xml"));
    }

    #[test]
    fn ragged_arrays_truncate_to_the_shortest() {
        let recent: SubmissionsRecent = serde_json::from_value(serde_json::json!({
            "accessionNumber": ["a", "b"],
            "filingDate": ["2026-01-01"],
            "form": ["4", "8-K"],
            "primaryDocument": ["x.xml", "y.htm"]
        }))
        .unwrap();
        assert_eq!(build_filings("1", &recent).len(), 1);
    }

    #[test]
    fn cik_lookup_matches_case_insensitively() {
        let body = serde_json::json!({
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"}
        });
        let want = "msft".to_uppercase();
        let mut found = None;
        for v in body.as_object().unwrap().values() {
            let row: TickerRow = serde_json::from_value(v.clone()).unwrap();
            if row.ticker.to_uppercase() == want {
                found = Some(format!("{:010}", row.cik_str));
            }
        }
        assert_eq!(found.as_deref(), Some("0000789019"));
    }

    /// Live checks. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to sec.gov"]
    async fn live_apple_filings() {
        let p = SecProvider::new();
        let cik = p.cik_for_ticker("AAPL").await.unwrap();
        assert_eq!(cik, "0000320193");
        let filings = p.fetch_sec_filings(&cik).await.unwrap();
        assert!(!filings.is_empty());
    }
}

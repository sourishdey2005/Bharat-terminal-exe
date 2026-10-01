// crates/bt-data/src/earnings.rs
// Author: Sourish Dey

//! Earnings calendar from Yahoo's `calendarEvents` module (no API key).
//!
//! Only the *date* of an earnings report is public here, never the result, so
//! the panel is a forward-looking calendar rather than a history. Dates arrive as
//! epoch seconds and are sometimes `None` for companies that have not announced,
//! which is why every field that can be absent is absent rather than zero: a
//! fabricated epoch date would render as 1970 and look like a real past event.

use bt_core::{BtError, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (compatible; BharatTerminal/4.2)";
const BASE_URL: &str = "https://query2.finance.yahoo.com/v10/finance/quoteSummary";

/// One scheduled (or just-reported) earnings event.
#[derive(Debug, Clone, PartialEq)]
pub struct EarningsEvent {
    /// Report date, or `None` when Yahoo has no announced date.
    pub date: Option<DateTime<Utc>>,
    pub symbol: String,
    /// Consensus EPS estimate. Absent rather than zero: "no estimate" and
    /// "estimated zero" are different facts.
    pub estimate_eps: Option<f64>,
    /// Reported EPS, when the event is already in the past.
    pub actual_eps: Option<f64>,
}

impl EarningsEvent {
    /// Whether this is still ahead of `now`.
    pub fn is_ahead(&self, now: DateTime<Utc>) -> bool {
        matches!(self.date, Some(d) if d > now)
    }

    /// Days until the event, negative if it has passed.
    pub fn days_until(&self, now: DateTime<Utc>) -> Option<i64> {
        self.date.map(|d| (d - now).num_days())
    }

    /// "in 3d", "today", "12d ago" — relative label for the calendar grid.
    pub fn relative_label(&self, now: DateTime<Utc>) -> String {
        match self.days_until(now) {
            None => "unscheduled".to_string(),
            Some(0) => "today".to_string(),
            Some(n) if n > 0 => format!("in {n}d"),
            Some(n) => format!("{}d ago", -n),
        }
    }
}

/// Parse the `calendarEvents` JSON for one symbol.
///
/// Tolerates the shape variants Yahoo emits in practice: the module can be
/// absent, the earnings dict can be absent, and both can be JSON `null`.
pub fn parse_calendar(symbol: &str, json: &str) -> Result<Vec<EarningsEvent>> {
    let root: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| BtError::DataFetch(format!("earnings parse: {e}")))?;

    // quoteSummary -> result[0] -> calendarEvents -> earnings
    let cal = &root["quoteSummary"]["result"][0]["calendarEvents"];
    let earnings = &cal["earnings"];
    let dates = &earnings["earningsDate"];
    if dates.is_null() {
        return Ok(Vec::new());
    }
    let arr = dates
        .as_array()
        .ok_or_else(|| BtError::DataFetch("earningsDate is not an array".into()))?;

    // Yahoo wraps these as {"raw": N, "fmt": "N"}; reading the object as a
    // bare f64 silently yields None, so the `.raw` is unwrapped explicitly.
    let actual = earnings["earningsAverage"]
        .get("raw")
        .and_then(|v| v.as_f64());
    let reported = earnings["reportedEarnings"]["reportedEPS"]["raw"].as_f64();

    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        // Each entry is {"raw": 1751...} or a bare number depending on version.
        let epoch = entry
            .get("raw")
            .and_then(|v| v.as_i64())
            .or_else(|| entry.as_i64());
        // Epoch 0 means "not announced" upstream, not 1 January 1970. Mapping
        // it through would render as an event 56 years in the past.
        let date = epoch
            .filter(|e| *e > 0)
            .and_then(|e| Utc.timestamp_opt(e, 0).single());
        out.push(EarningsEvent {
            date,
            symbol: symbol.to_string(),
            estimate_eps: actual,
            actual_eps: reported,
        });
    }
    out.sort_by_key(|e| e.date);
    Ok(out)
}

/// Parse a many-symbol map, dropping symbols that errored.
///
/// The per-symbol error is intentionally swallowed here: one company Yahoo will
/// not answer for must not blank the calendar for the other 49.
pub fn parse_calendar_batch(json: &str) -> Vec<EarningsEvent> {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(json) else {
        return Vec::new();
    };
    let Some(obj) = root.as_object() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (symbol, body) in obj {
        // Re-serialize just this symbol's subtree through the single parser so
        // there is exactly one place that understands Yahoo's shape.
        if let Ok(text) = serde_json::to_string(body) {
            if let Ok(events) = parse_calendar(symbol, &text) {
                out.extend(events);
            }
        }
    }
    out.sort_by_key(|e| (e.date, e.symbol.clone()));
    out
}

pub struct EarningsProvider {
    client: Client,
}

impl EarningsProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(20))
            .build()
            .expect("HTTP client");
        Self { client }
    }

    /// Earnings for one symbol.
    #[instrument(skip(self))]
    pub async fn fetch(&self, symbol: &str) -> Result<Vec<EarningsEvent>> {
        let url = format!(
            "{BASE_URL}/{}?modules=calendarEvents",
            crate::social::urlencode(symbol)
        );
        let text = self.get_text(&url).await?;
        parse_calendar(symbol, &text)
    }

    /// Earnings for many symbols, merging and sorting the result.
    ///
    /// Serial rather than `join_all`: this is called once per watchlist refresh,
    /// and a burst of ~40 simultaneous requests to Yahoo is what gets the app
    /// rate-limited (and the user shown an empty calendar).
    #[instrument(skip(self))]
    pub async fn fetch_many(&self, symbols: &[String]) -> Result<Vec<EarningsEvent>> {
        let mut out = Vec::new();
        for s in symbols {
            if let Ok(events) = self.fetch(s).await {
                out.extend(events);
            }
        }
        out.sort_by_key(|e| (e.date, e.symbol.clone()));
        Ok(out)
    }

    /// Upcoming events within `days` of `now`.
    ///
    /// Undated events are excluded: "upcoming" cannot include an unknown date.
    pub async fn fetch_upcoming(&self, symbols: &[String], days: i64) -> Result<Vec<EarningsEvent>> {
        let all = self.fetch_many(symbols).await?;
        let now = Utc::now();
        Ok(all
            .into_iter()
            .filter(|e| matches!(e.days_until(now), Some(d) if (0..=days).contains(&d)))
            .collect())
    }

    async fn get_text(&self, url: &str) -> Result<String> {
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(BtError::DataFetch(format!("earnings HTTP {status}")));
        }
        resp.text()
            .await
            .map_err(|e| BtError::DataFetch(e.to_string()))
    }
}

impl Default for EarningsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    /// 2026-03-15T00:00:00Z as an epoch second.
    const MAR_2026: i64 = 1_773_465_600;

    /// Build a `calendarEvents` payload.
    ///
    /// Uses placeholder substitution rather than `format!`: the fixture is
    /// almost entirely braces, and escaping them by hand is how a JSON fixture
    /// quietly ends up malformed.
    fn payload(epochs: &[i64], eps: Option<f64>) -> String {
        let dates = epochs
            .iter()
            .map(|e| format!(r#"{{"raw":{e}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let avg = match eps {
            Some(v) => format!(r#""earningsAverage":{{"raw":{v}}},"#),
            None => r#""earningsAverage":null,"#.to_string(),
        };
        format!(
            r#"{{"quoteSummary":{{"result":[{{"calendarEvents":{{"earnings":{{"earningsDate":[{dates}],{avg}"reportedEarnings":{{"reportedEPS":{{"raw":13.1}}}}}}}}}}]}}}}"#
        )
    }

    fn realistic(symbol: &str, epoch: i64) -> String {
        let _ = symbol;
        payload(&[epoch], Some(12.34))
    }

    #[test]
    fn parses_a_realistic_payload() {
        let evs = parse_calendar("RELIANCE.NS", &realistic("RELIANCE.NS", MAR_2026)).unwrap();
        assert_eq!(evs.len(), 1);
        let e = &evs[0];
        assert_eq!(e.symbol, "RELIANCE.NS");
        assert_eq!(e.estimate_eps, Some(12.34));
        assert_eq!(e.actual_eps, Some(13.1));
        assert_eq!(e.date.map(|d| d.timestamp()), Some(MAR_2026));
    }

    #[test]
    fn absent_earnings_is_an_empty_list_not_an_error() {
        // "No upcoming earnings" is a legitimate answer, not a failure.
        let json = r#"{"quoteSummary":{"result":[{"calendarEvents":{"earnings":{}}}]}}"#;
        assert!(parse_calendar("X", json).unwrap().is_empty());
        let null = r#"{"quoteSummary":{"result":[{"calendarEvents":{"earnings":{"earningsDate":null}}}]}}"#;
        assert!(parse_calendar("X", null).unwrap().is_empty());
    }

    #[test]
    fn an_unannounced_date_stays_none_never_epoch_zero() {
        // A zero/absent date must not become 1970-01-01, which would render as
        // a real event 56 years ago.
        let json = r#"{"quoteSummary":{"result":[{"calendarEvents":{"earnings":{
            "earningsDate":[{"raw":0}],"earningsAverage":{"raw":1.0}
        }}}]}}"#;
        let evs = parse_calendar("X", json).unwrap();
        assert_eq!(evs.len(), 1);
        assert!(evs[0].date.is_none(), "{:?}", evs[0]);
    }

    #[test]
    fn malformed_json_is_an_error() {
        assert!(parse_calendar("X", "").is_err());
        assert!(parse_calendar("X", "<html>rate limited</html>").is_err());
    }

    #[test]
    fn events_are_sorted_by_date() {
        let json = payload(&[MAR_2026 + 90_000, MAR_2026], Some(5.0));
        let evs = parse_calendar("X", &json).unwrap();
        let dates: Vec<i64> = evs.iter().filter_map(|e| e.date.map(|d| d.timestamp())).collect();
        assert_eq!(dates, vec![MAR_2026, MAR_2026 + 90_000], "must sort oldest-first");
    }

    #[test]
    fn relative_labels_read_correctly() {
        let now = Utc::now();
        let mk = |offset: i64| EarningsEvent {
            date: Some(now + ChronoDuration::days(offset)),
            symbol: "X".into(),
            estimate_eps: None,
            actual_eps: None,
        };
        assert_eq!(mk(3).relative_label(now), "in 3d");
        assert_eq!(mk(0).relative_label(now), "today");
        assert_eq!(mk(-12).relative_label(now), "12d ago");
        let undated = EarningsEvent {
            date: None,
            symbol: "X".into(),
            estimate_eps: None,
            actual_eps: None,
        };
        assert_eq!(undated.relative_label(now), "unscheduled");
        assert!(mk(3).is_ahead(now));
        assert!(!mk(-1).is_ahead(now));
    }

    #[test]
    fn missing_eps_stays_none_not_zero() {
        let json = r#"{"quoteSummary":{"result":[{"calendarEvents":{"earnings":{
            "earningsDate":[{"raw":1773465600}],"earningsAverage":null
        }}}]}}"#;
        let evs = parse_calendar("X", json).unwrap();
        assert_eq!(evs[0].estimate_eps, None, "no estimate must not read as an estimate of 0");
        assert_eq!(evs[0].actual_eps, None);
    }

    #[test]
    fn batch_parsing_merges_and_skips_broken_symbols() {
        let json = format!(
            r#"{{
              "AAPL": {},
              "BROKEN": "not-an-object",
              "TCS.NS": {}
            }}"#,
            realistic("AAPL", MAR_2026),
            realistic("TCS.NS", MAR_2026 + 86400)
        );
        let evs = parse_calendar_batch(&json);
        assert_eq!(evs.len(), 2, "{evs:?}");
        let syms: Vec<&str> = evs.iter().map(|e| e.symbol.as_str()).collect();
        assert!(syms.contains(&"AAPL") && syms.contains(&"TCS.NS"), "{syms:?}");
    }

    #[test]
    fn batch_parsing_of_garbage_yields_nothing() {
        assert!(parse_calendar_batch("not json").is_empty());
        assert!(parse_calendar_batch("[]").is_empty());
        assert!(parse_calendar_batch("{}").is_empty());
    }
}
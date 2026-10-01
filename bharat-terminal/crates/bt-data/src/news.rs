// crates/bt-data/src/news.rs
// Author: Sourish Dey

//! News headlines over RSS/Atom (no API key).
//!
//! `feed-rs` parses both RSS 2.0 and Atom, so one function serves exchange
//! feeds, Google News search feeds and central-bank press-release feeds.
//! Sentiment scoring lives in `bt_analytics::sentiment`, not here: fetching is
//! I/O, scoring is pure, and the two should stay separable for tests.

use bt_core::{BtError, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (compatible; BharatTerminal/4.0)";

/// Curated starting feeds. Google News search feeds need no key and cover
/// Indian markets well; the RBI press feed is the primary source for policy.
pub const DEFAULT_FEEDS: &[(&str, &str)] = &[
    (
        "Market India",
        "https://news.google.com/rss/search?q=NSE%20stock%20market&hl=en-IN&gl=IN&ceid=IN:en",
    ),
    (
        "RBI",
        "https://news.google.com/rss/search?q=RBI%20Reserve%20Bank%20of%20India&hl=en-IN&gl=IN&ceid=IN:en",
    ),
    (
        "Global Markets",
        "https://news.google.com/rss/search?q=stock%20market%20Sensex%20Nifty&hl=en-IN&gl=IN&ceid=IN:en",
    ),
];

/// One headline, ready for display and scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct NewsArticle {
    pub title: String,
    pub link: String,
    pub source: String,
    pub published: Option<DateTime<Utc>>,
    pub summary: String,
    /// VADER compound in [-1, 1], filled in by the caller via
    /// `bt_analytics::sentiment` (kept out of this crate so fetching stays I/O-only).
    pub sentiment: Option<f32>,
}

pub struct NewsProvider {
    client: Client,
}

impl NewsProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(20))
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    /// Fetch and parse one feed URL.
    #[instrument(skip(self))]
    pub async fn fetch_feed(&self, name: &str, url: &str) -> Result<Vec<NewsArticle>> {
        let bytes = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| BtError::DataFetch(format!("news network error ({name}): {e}")))?
            .error_for_status()
            .map_err(|e| BtError::DataFetch(format!("news HTTP error ({name}): {e}")))?
            .bytes()
            .await
            .map_err(|e| BtError::DataFetch(format!("news body error ({name}): {e}")))?;
        parse_feed_bytes(name, &bytes)
    }

    /// Fetch all [`DEFAULT_FEEDS`], skipping failures so one dead feed cannot
    /// take the whole News tab down. Newest first.
    #[instrument(skip(self))]
    pub async fn fetch_all(&self) -> Vec<NewsArticle> {
        let mut all = Vec::new();
        for (name, url) in DEFAULT_FEEDS {
            match self.fetch_feed(name, url).await {
                Ok(mut v) => all.append(&mut v),
                Err(e) => tracing::warn!("news feed {name} failed: {e}"),
            }
        }
        all.sort_by_key(|a| std::cmp::Reverse(a.published));
        all
    }
}

impl Default for NewsProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse raw feed bytes. Pure function, tested with fixtures.
pub fn parse_feed_bytes(source: &str, bytes: &[u8]) -> Result<Vec<NewsArticle>> {
    let feed = feed_rs::parser::parse(bytes)
        .map_err(|e| BtError::InvalidInput(format!("feed parse error ({source}): {e}")))?;
    Ok(feed
        .entries
        .into_iter()
        .filter_map(|e| {
            let title = e.title?.content.trim().to_string();
            if title.is_empty() {
                return None;
            }
            Some(NewsArticle {
                title,
                link: e.links.first().map(|l| l.href.clone()).unwrap_or_default(),
                source: source.to_string(),
                published: e.published,
                summary: e.summary.map(|s| s.content.trim().to_string()).unwrap_or_default(),
                sentiment: None,
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RSS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item><title>Reliance hits record high</title>
<link>https://example.com/1</link>
<description>Shares surged on strong results.</description>
<pubDate>Mon, 29 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title></title><link>https://example.com/2</link></item>
</channel></rss>"#;

    const ATOM: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Test</title>
<entry><title>RBI holds rates steady</title>
<link href="https://example.com/rbi"/>
<summary>Policy unchanged.</summary>
<updated>2026-09-29T10:00:00Z</updated></entry>
</feed>"#;

    #[test]
    fn parses_rss_items_and_skips_empty_titles() {
        let out = parse_feed_bytes("t", RSS.as_bytes()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].title, "Reliance hits record high");
        assert_eq!(out[0].link, "https://example.com/1");
        assert!(out[0].summary.contains("surged"));
        assert!(out[0].published.is_some());
    }

    #[test]
    fn parses_atom_entries() {
        let out = parse_feed_bytes("t", ATOM.as_bytes()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].title, "RBI holds rates steady");
        assert_eq!(out[0].link, "https://example.com/rbi");
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(parse_feed_bytes("t", b"not xml at all {{{").is_err());
        assert!(parse_feed_bytes("t", b"").is_err());
    }

    /// Live check. Run with: cargo test -p bt-data -- --ignored
    #[tokio::test]
    #[ignore = "requires live network access to news.google.com"]
    async fn live_feeds() {
        let p = NewsProvider::new();
        let all = p.fetch_all().await;
        assert!(!all.is_empty());
    }
}

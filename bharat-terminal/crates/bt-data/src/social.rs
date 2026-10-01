// crates/bt-data/src/social.rs
// Author: Sourish Dey

//! Free social sentiment from Reddit and StockTwits — no API keys.
//!
//! Both endpoints are public and unauthenticated, and both rate-limit hard
//! (HTTP 429 is routine on StockTwits from a shared IP). Every function here is
//! therefore best-effort: a failure returns an error the caller is expected to
//! surface as "sentiment unavailable", never to substitute a neutral score,
//! which would read as "nobody is talking about this" rather than "we could not
//! ask".
//!
//! The parsing is separated from the fetching ([`parse_reddit`],
//! [`parse_stocktwits`]) and unit-tested against captured fixture shapes. That
//! split is what lets the sentiment maths be tested without a network call, in a
//! suite that has to stay hermetic.

use bt_core::{BtError, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
const REDDIT_BASE: &str = "https://www.reddit.com";
const STOCKTWITS_BASE: &str = "https://api.stocktwits.com/api/2";

/// Subreddit searched per exchange.
pub fn subreddit_for(symbol: &str) -> &'static str {
    if symbol.ends_with(".NS") || symbol.ends_with(".BO") {
        "IndianStreetBets"
    } else if symbol.contains("BTC") || symbol.contains("ETH") {
        "CryptoCurrency"
    } else {
        "wallstreetbets"
    }
}

/// Bull/bear tally from one source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SentimentScore {
    /// Count of bullish-classified items.
    pub bullish: u32,
    /// Count of bearish-classified items.
    pub bearish: u32,
    /// Items that matched neither classifier. Reported so the UI can say
    /// "3 of 40 said anything", instead of implying a 100% sample.
    pub neutral: u32,
    /// Where the data came from, for display.
    pub source: &'static str,
}

impl SentimentScore {
    /// Items actually classified.
    pub fn total(&self) -> u32 {
        self.bullish + self.bearish
    }

    /// Net sentiment in `-1..=1`.
    ///
    /// Returns `None` when nothing was classified. A zero from an empty sample
    /// would be indistinguishable from a genuinely balanced crowd.
    pub fn net(&self) -> Option<f64> {
        let t = self.total();
        if t == 0 {
            return None;
        }
        Some((self.bullish as f64 - self.bearish as f64) / t as f64)
    }

    /// Combine several sources by summing the tallies.
    ///
    /// Summing rather than averaging the per-source scores keeps a source with
    /// 10 posts from cancelling one with 1000.
    pub fn merge(parts: &[SentimentScore]) -> SentimentScore {
        let bullish = parts.iter().map(|p| p.bullish).sum();
        let bearish = parts.iter().map(|p| p.bearish).sum();
        let neutral = parts.iter().map(|p| p.neutral).sum();
        SentimentScore {
            bullish,
            bearish,
            neutral,
            source: "combined",
        }
    }

    /// An empty score for `source`.
    pub fn empty(source: &'static str) -> Self {
        SentimentScore {
            bullish: 0,
            bearish: 0,
            neutral: 0,
            source,
        }
    }
}

/// Multi-word phrases that mean the opposite of the word they contain.
///
/// "short squeeze" is bullish even though "short" is a bearish keyword, and a
/// lexicon that reads it as -1 would score one of the most bullish events in
/// retail trading as bearish. Phrases are checked before individual words for
/// exactly this reason.
const BEARISH_CONTAINING_BULLISH: &[&str] = &["short squeeze", "bear trap", "bull run"];
const BULLISH_CONTAINING_BEARISH: &[&str] = &["bull trap", "bear rally"];

/// Classify free text as bullish, bearish, or neither.
///
/// A small hand-written lexicon rather than a model: it is deterministic, has no
/// vocabulary file to ship, and its failures are readable. Negation is applied
/// as a whole-word match and nets off before it is applied, so "not a bubble"
/// cannot be counted as bearish *and* as negation of bearish.
pub fn classify(text: &str) -> i8 {
    let t = text.to_lowercase();

    // Idiom first: these are not keyword soup.
    for p in BEARISH_CONTAINING_BULLISH {
        if t.contains(p) {
            return 1;
        }
    }
    for p in BULLISH_CONTAINING_BEARISH {
        if t.contains(p) {
            return -1;
        }
    }

    let bullish = [
        "bull", "long", "buy", "calls", "upside", "moon", "squeeze", "breakout", "undervalued",
    ];
    let bearish = [
        "bear", "short", "sell", "puts", "downside", "dump", "overvalued", "bubble", "crash",
    ];

    let count = |terms: &[&str]| terms.iter().filter(|w| t.contains(*w)).count();
    let b = count(&bullish) as i8;
    let s = count(&bearish) as i8;
    if b == 0 && s == 0 {
        return 0;
    }

    let net = b - s;
    // Negated once, to the netted result, and matched as a standalone word so a
    // trailing "not" ("definitely not") counts as clearly as "not bullish".
    if has_negation(&t) {
        -net
    } else {
        net
    }
}

/// Whether the lowercased text carries a standalone negator.
///
/// Whole-word matching is required: `"not"` as a substring fires inside
/// "nothing", which would flip the meaning of half the corpus.
fn has_negation(t: &str) -> bool {
    const NEGATORS: &[&str] = &[
        "not", "no", "never", "isn't", "isnt", "won't", "wont", "don't", "dont", "anti",
    ];
    t.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '\'')
        .any(|w| NEGATORS.contains(&w))
}

#[derive(Debug, Deserialize)]
struct RedditPost {
    title: Option<String>,
    #[serde(rename = "selftext")]
    body: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RedditChild {
    data: RedditPost,
}

/// Reddit `/search.json` nests exactly two levels: `{ "data": { "children":
/// [ { "data": {title, selftext} } ] } }`. A third wrapper here would make every
/// real response fail to parse.
#[derive(Debug, Deserialize)]
struct RedditListing {
    children: Vec<RedditChild>,
}

#[derive(Debug, Deserialize)]
struct RedditSearch {
    data: RedditListing,
}

#[derive(Debug, Deserialize)]
struct TwMessage {
    body: String,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct TwStream {
    messages: Vec<TwMessage>,
}

/// Parse a Reddit `/search.json` payload.
///
/// Unparseable input is an error, not an empty score: the difference between
/// "Reddit said nothing" and "Reddit's shape changed" is the difference between
/// an honest blank panel and a silently wrong one.
pub fn parse_reddit(json: &str) -> Result<SentimentScore> {
    let parsed: RedditSearch = serde_json::from_str(json)
        .map_err(|e| BtError::DataFetch(format!("reddit search parse: {e}")))?;
    let mut out = SentimentScore::empty("Reddit");
    for child in parsed.data.children {
        let text = match (child.data.title, child.data.body) {
            (Some(t), Some(b)) => format!("{t} {b}"),
            (Some(t), None) => t,
            (None, Some(b)) => b,
            (None, None) => continue,
        };
        match classify(&text) {
            v if v > 0 => out.bullish += 1,
            v if v < 0 => out.bearish += 1,
            _ => out.neutral += 1,
        }
    }
    Ok(out)
}

/// Parse a StockTwits `/streams/symbol/{sym}.json` payload.
pub fn parse_stocktwits(json: &str) -> Result<SentimentScore> {
    let parsed: TwStream = serde_json::from_str(json)
        .map_err(|e| BtError::DataFetch(format!("stocktwits parse: {e}")))?;
    let mut out = SentimentScore::empty("StockTwits");
    for msg in parsed.messages {
        match classify(&msg.body) {
            v if v > 0 => out.bullish += 1,
            v if v < 0 => out.bearish += 1,
            _ => out.neutral += 1,
        }
    }
    Ok(out)
}

pub struct SocialProvider {
    client: Client,
}

impl SocialProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(15))
            .build()
            .expect("HTTP client");
        Self { client }
    }

    /// Fetch Reddit sentiment for `symbol`.
    #[instrument(skip(self))]
    pub async fn fetch_reddit(&self, symbol: &str) -> Result<SentimentScore> {
        let url = format!(
            "{REDDIT_BASE}/r/{}/search.json?q={}&restrict_sr=1&sort=new&limit=100",
            subreddit_for(symbol),
            urlencode(symbol)
        );
        let text = self
            .get_text(&url)
            .await
            .map_err(|e| BtError::DataFetch(format!("reddit: {e}")))?;
        parse_reddit(&text)
    }

    /// Fetch StockTwits sentiment for `symbol`.
    ///
    /// StockTwits uses bare tickers: `RELIANCE.NS` becomes `RELIANCE`.
    #[instrument(skip(self))]
    pub async fn fetch_stocktwits(&self, symbol: &str) -> Result<SentimentScore> {
        let bare = symbol.split('.').next().unwrap_or(symbol);
        let url = format!("{STOCKTWITS_BASE}/streams/symbol/{}.json", urlencode(bare));
        let text = self
            .get_text(&url)
            .await
            .map_err(|e| BtError::DataFetch(format!("stocktwits: {e}")))?;
        parse_stocktwits(&text)
    }

    /// Both sources merged. Either one failing is tolerated as long as one
    /// answered; both failing propagates the error.
    #[instrument(skip(self))]
    pub async fn fetch_combined(&self, symbol: &str) -> Result<SentimentScore> {
        let (reddit, tw) = tokio::join!(self.fetch_reddit(symbol), self.fetch_stocktwits(symbol));
        let mut parts = Vec::new();
        if let Ok(r) = reddit {
            parts.push(r);
        }
        if let Ok(t) = tw {
            parts.push(t);
        }
        if parts.is_empty() {
            return Err(BtError::DataFetch(
                "both social sources failed".to_string(),
            ));
        }
        Ok(SentimentScore::merge(&parts))
    }

    async fn get_text(&self, url: &str) -> std::result::Result<String, String> {
        let resp = self.client.get(url).send().await.map_err(|e| e.to_string())?;
        let status = resp.status();
        if !status.is_success() {
            // 429 is the common case and deserves its own wording, because
            // "rate limited" and "does not exist" call for different retries.
            if status.as_u16() == 429 {
                return Err(format!("rate limited (429), try again shortly"));
            }
            return Err(format!("HTTP {status}"));
        }
        resp.text().await.map_err(|e| e.to_string())
    }
}

impl Default for SocialProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Percent-encode the characters that actually appear in tickers and queries.
///
/// Shared rather than duplicated per provider: two modules encoding `.` one way
/// and `%` another is the kind of thing that only breaks in production.
pub fn urlencode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "+".to_string(),
            other => other
                .to_string()
                .bytes()
                .map(|b| format!("%{b:02X}"))
                .collect(),
        })
        .collect()
}

/// One classified social post, for the timeline views.
#[derive(Debug, Clone, PartialEq)]
pub struct SentimentPoint {
    pub at: DateTime<Utc>,
    pub symbol: String,
    /// Net sentiment at this post.
    pub score: f64,
    pub source: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_finds_the_obvious_terms() {
        assert!(classify("this is going to the moon") > 0);
        assert!(classify("BUY the dip") > 0);
        assert!(classify("short squeeze incoming") > 0);
        assert!(classify("total dump, exit now") < 0);
        assert!(classify("puts printing") < 0);
    }

    #[test]
    fn classify_is_neutral_on_unrelated_text() {
        assert_eq!(classify("meeting at 3pm"), 0);
        assert_eq!(classify(""), 0);
        assert_eq!(classify("the weather is fine"), 0);
    }

    #[test]
    fn negation_flips_the_read() {
        // "not bullish" scoring as bullish is worse than not scoring.
        assert!(classify("not bullish at all") < 0);
        assert!(classify("this is a bubble, definitely not") > 0);
    }

    #[test]
    fn classify_never_overflows_i8() {
        // Every term present in both lists nets to zero, and no input may push
        // the counter past i8.
        let text = format!("{} {}", "bull ".repeat(200), "bear ".repeat(200));
        let v = classify(&text);
        assert!(v >= -1 && v <= 1, "got {v}");
    }

    #[test]
    fn net_is_none_for_an_empty_sample() {
        let s = SentimentScore::empty("test");
        assert_eq!(s.net(), None, "no posts must not read as neutral sentiment");
    }

    #[test]
    fn net_ignores_neutral_items_in_the_denominator() {
        // 3 bull, 1 bear, 40 unclassified -> 0.5, not 0.09.
        let s = SentimentScore {
            bullish: 3,
            bearish: 1,
            neutral: 40,
            source: "test",
        };
        assert_eq!(s.total(), 4);
        assert!((s.net().unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn merge_sums_rather_than_averaging() {
        let big = SentimentScore {
            bullish: 100,
            bearish: 0,
            neutral: 0,
            source: "big",
        };
        let small = SentimentScore {
            bullish: 0,
            bearish: 10,
            neutral: 0,
            source: "small",
        };
        let m = SentimentScore::merge(&[big, small]);
        assert_eq!(m.bullish, 100);
        assert_eq!(m.bearish, 10);
        assert!((m.net().unwrap() - 90.0 / 110.0).abs() < 1e-12, "a small source must not cancel a large one");
    }

    #[test]
    fn parse_reddit_classifies_a_realistic_payload() {
        let json = r#"{
          "data": { "children": [
            { "data": { "title": "RELIANCE to the moon", "selftext": "long calls" } },
            { "data": { "title": "total dump", "selftext": "" } },
            { "data": { "title": "meeting notes", "selftext": "" } }
          ]}
        }"#;
        let s = parse_reddit(json).expect("parse");
        assert_eq!(s.bullish, 1);
        assert_eq!(s.bearish, 1);
        assert_eq!(s.neutral, 1);
        assert_eq!(s.source, "Reddit");
    }

    #[test]
    fn parse_reddit_survives_missing_optional_fields() {
        let json = r#"{"data":{"children":[
            {"data":{"title":"bull run"}},
            {"data":{"selftext":"bear case"}},
            {"data":{"title":"x","selftext":"y"}}
        ]}}"#;
        let s = parse_reddit(json).expect("parse");
        assert_eq!(s.bullish, 1);
        assert_eq!(s.bearish, 1);
    }

    #[test]
    fn parse_reddit_rejects_a_changed_shape_rather_than_reporting_zero() {
        // The whole point of parsing separately: an HTML error page or a
        // changed schema must be visible, not silently read as "no posts".
        assert!(parse_reddit("<html>429 Too Many Requests</html>").is_err());
        assert!(parse_reddit("{}").is_err());
        assert!(parse_reddit("").is_err());
    }

    #[test]
    fn parse_reddit_handles_an_empty_listing() {
        let s = parse_reddit(r#"{"data":{"children":[]}}"#).expect("parse");
        assert_eq!(s.total(), 0);
        assert_eq!(s.net(), None);
    }

    #[test]
    fn parse_stocktwits_classifies_a_realistic_payload() {
        let json = r#"{"messages":[
            {"body":"BUY RELIANCE breakout","created_at":"2026-01-01T00:00:00Z"},
            {"body":"puts are printing","created_at":"2026-01-01T00:01:00Z"},
            {"body":"no opinion","created_at":"2026-01-01T00:02:00Z"}
        ]}"#;
        let s = parse_stocktwits(json).expect("parse");
        assert_eq!(s.bullish, 1);
        assert_eq!(s.bearish, 1);
        assert_eq!(s.neutral, 1);
        assert_eq!(s.source, "StockTwits");
    }

    #[test]
    fn parse_stocktwits_rejects_a_bad_shape() {
        assert!(parse_stocktwits("not json").is_err());
        assert!(parse_stocktwits(r#"{"nope":1}"#).is_err());
    }

    #[test]
    fn subreddit_selection_follows_the_listing() {
        assert_eq!(subreddit_for("RELIANCE.NS"), "IndianStreetBets");
        assert_eq!(subreddit_for("BTC-USD"), "CryptoCurrency");
        assert_eq!(subreddit_for("AAPL"), "wallstreetbets");
    }

    #[test]
    fn urlencode_escapes_the_awkward_characters() {
        assert_eq!(urlencode("RELIANCE.NS"), "RELIANCE.NS");
        assert_eq!(urlencode("BRK/B"), "BRK%2FB");
        assert_eq!(urlencode("a b"), "a+b");
        assert_eq!(urlencode("100%"), "100%25");
    }
}
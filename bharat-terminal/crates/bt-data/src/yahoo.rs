// crates/bt-data/src/yahoo.rs
// Author: Sourish Dey

//! Yahoo Finance direct HTTP provider (no external crate required).
//! Fetches real OHLCV data, quotes, search results, and company profiles
//! from Yahoo Finance's public endpoints.

use crate::provider::{CompanyProfile, DataProvider, Interval, Quote, SymbolInfo};
use bt_core::{BtError, Candle, OhlcvSeries, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration as StdDuration;
use tracing::instrument;

const USER_AGENT: &str = "Mozilla/5.0 (compatible; BharatTerminal/2.0)";
const BASE_CHART: &str = "https://query1.finance.yahoo.com/v8/finance/chart";
const BASE_QUOTE: &str = "https://query1.finance.yahoo.com/v7/finance/quote";
const BASE_SEARCH: &str = "https://query2.finance.yahoo.com/v1/finance/search";
const BASE_PROFILE: &str = "https://query1.finance.yahoo.com/v10/finance/quoteSummary";

/// Longest span Yahoo will serve at 1-minute granularity (~30 days).
const MIN1_MAX_DAYS: i64 = 30;
/// Longest span Yahoo will serve at 5-minute granularity (~60 days).
const MIN5_MAX_DAYS: i64 = 60;
/// Longest span Yahoo will serve at 15/30-minute granularity (~60 days).
const MIN15_MAX_DAYS: i64 = 60;
/// Longest span Yahoo will serve hourly, and the longest span any intraday
/// granularity supports at all (~730 days). Beyond this Yahoo answers
/// "Unsupported granularity" no matter what.
const HOUR1_MAX_DAYS: i64 = 730;

/// Coarsen an interval to the finest one Yahoo can serve over the requested
/// span.
///
/// Yahoo's chart API pairs each intraday granularity with a maximum history.
/// Asking for 1-minute bars over five years is not "a long window", it is an
/// impossible request, and it comes back as HTTP 422
/// `{"message":"Unsupported granularity"}`. The app hit this whenever a custom
/// date range crossed years while an intraday interval was selected.
///
/// Rather than fail, the interval steps up to daily once the window exceeds
/// what the requested granularity supports. Daily and coarser intervals are left
/// untouched, and intraday requests inside their window keep full resolution.
///
/// The preset `range` is used when there is no explicit timestamp window; the
/// longest presets are longer than any intraday window, so those coarsen too.
fn coarsen_interval(
    interval: Interval,
    period1: Option<i64>,
    period2: Option<i64>,
    range: &str,
) -> Interval {
    let days = match (period1, period2) {
        (Some(p1), Some(p2)) => ((p2 - p1) / 86_400).max(0),
        _ => range_days(range),
    };

    let max_days = match interval {
        Interval::Min1 => MIN1_MAX_DAYS,
        Interval::Min5 => MIN5_MAX_DAYS,
        Interval::Min15 | Interval::Min30 => MIN15_MAX_DAYS,
        Interval::Hour1 => HOUR1_MAX_DAYS,
        // Daily and coarser are always available, whatever the span.
        _ => return interval,
    };

    if days <= max_days {
        interval
    } else {
        Interval::Day1
    }
}

/// Approximate span of a Yahoo preset range string, in days.
fn range_days(range: &str) -> i64 {
    match range {
        "1d" | "5d" => 5,
        "1mo" => 30,
        "3mo" => 90,
        "6mo" => 180,
        "1y" => 365,
        "2y" => 730,
        "5y" | "10y" | "ytd" | "max" => 3650,
        _ => 365,
    }
}

#[derive(Debug, Deserialize)]
struct YahooChartResponse {
    chart: ChartData,
}

#[derive(Debug, Deserialize)]
struct ChartData {
    result: Option<Vec<ChartResult>>,
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChartResult {
    timestamp: Option<Vec<i64>>,
    indicators: Indicators,
    meta: ChartMeta,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChartMeta {
    symbol: String,
    #[serde(rename = "regularMarketPrice")]
    regular_market_price: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Indicators {
    quote: Vec<QuoteIndicators>,
}

#[derive(Debug, Deserialize)]
struct QuoteIndicators {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct YahooQuoteResponse {
    #[serde(rename = "quoteResponse")]
    quote_response: QuoteResponse,
}

#[derive(Debug, Deserialize)]
struct QuoteResponse {
    result: Vec<QuoteResult>,
}

#[derive(Debug, Deserialize)]
struct QuoteResult {
    symbol: String,
    #[serde(rename = "regularMarketPrice")]
    regular_market_price: Option<f64>,
    #[serde(rename = "regularMarketChange")]
    regular_market_change: Option<f64>,
    #[serde(rename = "regularMarketChangePercent")]
    regular_market_change_percent: Option<f64>,
    #[serde(rename = "regularMarketVolume")]
    regular_market_volume: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct YahooSearchResponse {
    quotes: Vec<SearchQuote>,
}

#[derive(Debug, Deserialize)]
struct SearchQuote {
    symbol: String,
    #[serde(rename = "shortname")]
    short_name: Option<String>,
    #[serde(rename = "longname")]
    long_name: Option<String>,
    exchange: Option<String>,
    #[serde(rename = "quoteType")]
    quote_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct YahooProfileResponse {
    #[serde(rename = "quoteSummary")]
    quote_summary: QuoteSummary,
}

#[derive(Debug, Deserialize)]
struct QuoteSummary {
    result: Vec<ProfileResult>,
}

#[derive(Debug, Deserialize)]
struct ProfileResult {
    #[serde(rename = "longName")]
    long_name: Option<String>,
    sector: Option<String>,
    industry: Option<String>,
    #[serde(rename = "marketCap")]
    market_cap: Option<MarketCap>,
    #[serde(rename = "fullTimeEmployees")]
    full_time_employees: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct MarketCap {
    raw: u64,
}

/// Yahoo Finance provider using direct HTTP calls.
pub struct YahooProvider {
    client: Client,
}

impl YahooProvider {
    /// Create a new Yahoo Finance provider.
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(StdDuration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    fn build_chart_url(
        &self,
        symbol: &str,
        interval: Interval,
        range: &str,
        period1: Option<i64>,
        period2: Option<i64>,
    ) -> String {
        // Yahoo accepts *either* a preset `range` *or* explicit `period1` /
        // `period2` timestamps. When both are sent, the preset wins and the
        // timestamps are silently ignored -- which capped every long custom
        // window at 5 years of data. So an explicit window is built without
        // `range` at all.
        match (period1, period2) {
            (Some(p1), Some(p2)) => format!(
                "{}/{}?interval={}&period1={}&period2={}",
                BASE_CHART,
                symbol,
                interval.as_str(),
                p1,
                p2
            ),
            _ => format!(
                "{}/{}?interval={}&range={}",
                BASE_CHART,
                symbol,
                interval.as_str(),
                range
            ),
        }
    }

    async fn fetch_with_retry(&self, url: &str) -> Result<reqwest::Response> {
        let mut attempts = 0;
        const MAX_RETRIES: u32 = 3;

        loop {
            let resp = self
                .client
                .get(url)
                .send()
                .await
                .map_err(|e| BtError::InvalidInput(format!("Network error: {}", e)))?;

            if resp.status() == 429 {
                attempts += 1;
                if attempts >= MAX_RETRIES {
                    return Err(BtError::InvalidInput(
                        "Rate limited (429) after 3 retries".to_string(),
                    ));
                }
                let delay = StdDuration::from_millis(500 * 2_u64.pow(attempts - 1));
                tokio::time::sleep(delay).await;
                continue;
            }

            if !resp.status().is_success() {
                return Err(BtError::InvalidInput(format!(
                    "HTTP {}: {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                )));
            }

            return Ok(resp);
        }
    }

    /// Fetch OHLCV history for a symbol.
    pub async fn fetch_ohlcv_range(
        &self,
        symbol: &str,
        interval: Interval,
        range: &str,
        period1: Option<i64>,
        period2: Option<i64>,
    ) -> Result<OhlcvSeries> {
        // Yahoo rejects an intraday interval combined with a long `range`, and
        // rejects very long intraday windows outright:
        //
        //     interval=1h&range=5y   -> 422 "Unsupported granularity"
        //     interval=1h&range=60d  -> 200
        //     interval=1d&range=5y   -> 200
        //
        // A custom date range can legitimately ask for hourly bars over years,
        // which is what produced that error. Rather than surface it, the
        // granularity is coarsened to the finest one Yahoo will actually serve
        // for the requested span, so the user gets bars instead of an error.
        let interval = coarsen_interval(interval, period1, period2, range);
        let url = self.build_chart_url(symbol, interval, range, period1, period2);
        let resp = self.fetch_with_retry(&url).await?;
        let data: YahooChartResponse = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("JSON parse error: {}", e)))?;

        let result = data
            .chart
            .result
            .and_then(|r| r.into_iter().next())
            .ok_or_else(|| BtError::EmptySeries(symbol.to_string()))?;

        if let Some(error) = data.chart.error {
            return Err(BtError::InvalidInput(format!("Yahoo error: {}", error)));
        }

        let timestamps = result.timestamp.unwrap_or_default();
        let quote = result
            .indicators
            .quote
            .into_iter()
            .next()
            .ok_or_else(|| BtError::EmptySeries(symbol.to_string()))?;

        let mut candles = Vec::with_capacity(timestamps.len());

        for (i, &ts) in timestamps.iter().enumerate() {
            let open = quote.open.get(i).and_then(|v| *v);
            let high = quote.high.get(i).and_then(|v| *v);
            let low = quote.low.get(i).and_then(|v| *v);
            let close = quote.close.get(i).and_then(|v| *v);
            let volume = quote.volume.get(i).and_then(|v| *v).unwrap_or(0.0);

            if let (Some(o), Some(h), Some(l), Some(c)) = (open, high, low, close) {
                candles.push(Candle::new(ts as f64, o, h, l, c, volume));
            }
        }

        if candles.is_empty() {
            return Err(BtError::EmptySeries(symbol.to_string()));
        }

        Ok(OhlcvSeries::new(symbol, candles))
    }
}

#[async_trait::async_trait]
impl DataProvider for YahooProvider {
    #[instrument(skip(self))]
    async fn fetch_ohlcv(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<OhlcvSeries> {
        let days = (end - start).num_days();
        let range = if days <= 1 {
            "1d".to_string()
        } else if days <= 5 {
            "5d".to_string()
        } else if days <= 30 {
            "1mo".to_string()
        } else if days <= 90 {
            "3mo".to_string()
        } else if days <= 180 {
            "6mo".to_string()
        } else if days <= 365 {
            "1y".to_string()
        } else if days <= 730 {
            "2y".to_string()
        } else {
            "5y".to_string()
        };

        // For ranges longer than 5 years, use explicit timestamps to bypass
        // Yahoo's 5y limit on the range parameter.
        let (period1, period2) = if days > 1825 {
            (Some(start.timestamp()), Some(end.timestamp()))
        } else {
            (None, None)
        };

        self.fetch_ohlcv_range(symbol, interval, &range, period1, period2)
            .await
    }

    #[instrument(skip(self))]
    async fn fetch_quote(&self, symbol: &str) -> Result<Quote> {
        let url = format!("{}?symbols={}", BASE_QUOTE, symbol);
        let resp = self.fetch_with_retry(&url).await?;
        let data: YahooQuoteResponse = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("JSON parse error: {}", e)))?;

        let result = data
            .quote_response
            .result
            .into_iter()
            .next()
            .ok_or_else(|| BtError::EmptySeries(symbol.to_string()))?;

        Ok(Quote {
            symbol: result.symbol,
            price: result.regular_market_price.unwrap_or(0.0),
            change: result.regular_market_change.unwrap_or(0.0),
            change_pct: result.regular_market_change_percent.unwrap_or(0.0),
            volume: result.regular_market_volume.unwrap_or(0),
            timestamp: Utc::now(),
        })
    }

    #[instrument(skip(self))]
    async fn search_symbols(&self, query: &str) -> Result<Vec<SymbolInfo>> {
        let url = format!("{}?q={}&quotesCount=20", BASE_SEARCH, query);
        let resp = self.fetch_with_retry(&url).await?;
        let data: YahooSearchResponse = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("JSON parse error: {}", e)))?;

        let results = data
            .quotes
            .into_iter()
            .map(|q| SymbolInfo {
                symbol: q.symbol,
                name: q.long_name.or(q.short_name).unwrap_or_default(),
                exchange: q.exchange.unwrap_or_default(),
                asset_type: q.quote_type.unwrap_or_default(),
            })
            .collect();

        Ok(results)
    }

    #[instrument(skip(self))]
    async fn fetch_company_profile(&self, symbol: &str) -> Result<CompanyProfile> {
        let url = format!(
            "{}/{}?modules=assetProfile,summaryDetail",
            BASE_PROFILE, symbol
        );
        let resp = self.fetch_with_retry(&url).await?;
        let data: YahooProfileResponse = resp
            .json()
            .await
            .map_err(|e| BtError::InvalidInput(format!("JSON parse error: {}", e)))?;

        let result = data
            .quote_summary
            .result
            .into_iter()
            .next()
            .ok_or_else(|| BtError::EmptySeries(symbol.to_string()))?;

        Ok(CompanyProfile {
            symbol: symbol.to_string(),
            name: result.long_name.unwrap_or_default(),
            sector: result.sector.unwrap_or_default(),
            industry: result.industry.unwrap_or_default(),
            market_cap: result.market_cap.map(|m| m.raw as f64).unwrap_or(0.0),
            employees: result.full_time_employees.unwrap_or(0),
        })
    }
}

impl Default for YahooProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resolve_indian_symbol() {
        let provider = YahooProvider::new();
        let series = provider
            .fetch_ohlcv_range("RELIANCE.NS", Interval::Day1, "1mo", None, None)
            .await;
        assert!(series.is_ok());
        let s = series.unwrap();
        assert!(!s.candles.is_empty());
        assert_eq!(s.symbol, "RELIANCE.NS");
    }

    #[tokio::test]
    async fn test_resolve_us_symbol() {
        let provider = YahooProvider::new();
        let series = provider
            .fetch_ohlcv_range("AAPL", Interval::Day1, "1mo", None, None)
            .await;
        assert!(series.is_ok());
        let s = series.unwrap();
        assert!(!s.candles.is_empty());
        assert_eq!(s.symbol, "AAPL");
    }

    #[test]
    fn test_chart_url_with_periods_omits_range() {
        // Regression test: sending `range=5y` alongside `period1`/`period2`
        // made Yahoo honour the preset and silently drop the timestamps, so a
        // 2017-2026 custom window only ever charted ~5 years ending now.
        let provider = YahooProvider::new();
        let url = provider.build_chart_url(
            "RELIANCE.NS",
            Interval::Week1,
            "5y",
            Some(1_505_000_000),
            Some(1_790_000_000),
        );
        assert!(
            url.contains("period1=1505000000"),
            "start timestamp missing: {url}"
        );
        assert!(
            url.contains("period2=1790000000"),
            "end timestamp missing: {url}"
        );
        assert!(
            !url.contains("range="),
            "preset range must not shadow explicit timestamps: {url}"
        );
        assert!(url.contains("interval=1wk"), "weekly interval: {url}");
    }

    #[test]
    fn test_chart_url_without_periods_uses_range() {
        let provider = YahooProvider::new();
        let url = provider.build_chart_url("AAPL", Interval::Day1, "1mo", None, None);
        assert!(url.contains("range=1mo"), "preset range: {url}");
        assert!(!url.contains("period1"), "no timestamps expected: {url}");
        assert!(!url.contains("period2"), "no timestamps expected: {url}");
    }

    #[test]
    fn test_chart_url_partial_periods_fall_back_to_range() {
        // A lone timestamp cannot define a window, so it must be dropped
        // rather than sent half-formed.
        let provider = YahooProvider::new();
        let url =
            provider.build_chart_url("AAPL", Interval::Day1, "1mo", Some(1_700_000_000), None);
        assert!(url.contains("range=1mo"), "preset range: {url}");
        assert!(!url.contains("period1"), "half-formed window: {url}");
    }

    #[tokio::test]
    async fn test_search() {
        let provider = YahooProvider::new();
        let results = provider.search_symbols("RELIANCE").await;
        assert!(results.is_ok());
        let r = results.unwrap();
        assert!(!r.is_empty());
    }

    /// Yahoo caps each intraday granularity to a maximum history. A custom
    /// window crossing years with an intraday interval used to come back as
    /// HTTP 422 "Unsupported granularity"; the interval must be coarsened
    /// instead of failing.
    #[test]
    fn test_intraday_interval_coarsens_past_yahoos_window() {
        let six_years_days = 2192i64;
        let p1 = 1_577_836_800i64;
        let p2 = p1 + six_years_days * 86_400;

        // Reproduces the 422 exactly; all of these must coarsen to daily.
        for iv in [
            Interval::Min1,
            Interval::Min5,
            Interval::Min15,
            Interval::Min30,
            Interval::Hour1,
        ] {
            assert_eq!(
                coarsen_interval(iv, Some(p1), Some(p2), "5y"),
                Interval::Day1,
                "{iv:?} over {six_years_days}d should coarsen to daily"
            );
        }
    }

    #[test]
    fn test_intraday_interval_is_preserved_inside_its_window() {
        let p1 = 1_700_000_000i64;
        // 1-minute data is only kept for ~30 days.
        let within = p1 + 20 * 86_400;
        assert_eq!(
            coarsen_interval(Interval::Min1, Some(p1), Some(within), "1mo"),
            Interval::Min1
        );
        // Just past the limit it must coarsen.
        let beyond = p1 + 45 * 86_400;
        assert_eq!(
            coarsen_interval(Interval::Min1, Some(p1), Some(beyond), "1mo"),
            Interval::Day1
        );
    }

    #[test]
    fn test_hourly_survives_a_years_but_not_longer() {
        let p1 = 1_577_836_800i64;
        // Hourly reaches back ~2 years.
        let one_year = p1 + 365 * 86_400;
        assert_eq!(
            coarsen_interval(Interval::Hour1, Some(p1), Some(one_year), "1y"),
            Interval::Hour1
        );
        let three_years = p1 + 1095 * 86_400;
        assert_eq!(
            coarsen_interval(Interval::Hour1, Some(p1), Some(three_years), "5y"),
            Interval::Day1
        );
    }

    #[test]
    fn test_daily_and_coarser_are_never_coarsened() {
        let p1 = 1_577_836_800i64;
        let p2 = p1 + 2192 * 86_400;
        for iv in [Interval::Day1, Interval::Week1, Interval::Month1] {
            assert_eq!(
                coarsen_interval(iv, Some(p1), Some(p2), "5y"),
                iv,
                "{iv:?} is available at any span"
            );
        }
    }

    #[test]
    fn test_preset_ranges_are_measured_without_timestamps() {
        // No explicit window: the preset range still has to coarsen, because
        // a five-year preset with an intraday interval is a 422 just the same.
        assert_eq!(
            coarsen_interval(Interval::Hour1, None, None, "5y"),
            Interval::Day1
        );
        assert_eq!(
            coarsen_interval(Interval::Min5, None, None, "1d"),
            Interval::Min5
        );
        assert_eq!(range_days("5y"), 3650);
        assert_eq!(range_days("60d"), 365);
    }
}

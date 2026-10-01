// crates/bt-data/src/lib.rs
// Author: Sourish Dey

//! bt-data: Real market data providers for Bharat Terminal.
//!
//! Provides access to free public market data APIs:
//! - Yahoo Finance (stocks, ETFs, indices, crypto, FX)
//! - Coinbase (crypto spot markets)
//! - SQLite caching with TTL-based expiration
//! - Symbol resolution for 50+ major companies

pub mod bhavcopy;
pub mod binance;
pub mod cache;
pub mod coinbase;
pub mod fmp;
pub mod india;
pub mod moneycontrol;
pub mod news;
pub mod nse;
pub mod provider;
pub mod sec;
pub mod symbol;
pub mod worldbank;
pub mod yahoo;
pub mod amfi;

pub use provider::{CompanyProfile, DataProvider, Interval, Quote, SymbolInfo};
pub use symbol::{COMPANY_LIST, DEFAULT_COMPANY};

use crate::bhavcopy::BhavcopyProvider;
use crate::cache::Cache;
use crate::coinbase::CoinbaseProvider;
use crate::yahoo::YahooProvider;
use bt_core::{BtError, OhlcvSeries, Result};
use chrono::{DateTime, Utc};
use std::path::Path;

/// High-level data service combining providers with caching.
pub struct DataService {
    yahoo: YahooProvider,
    coinbase: CoinbaseProvider,
    /// Official exchange settlement files, used when Yahoo is unavailable.
    bhavcopy: BhavcopyProvider,
    cache: Cache,
}

/// Directory that holds the on-disk cache and settings.
///
/// Anchored to the executable's own directory rather than the working
/// directory: a relative `./data` resolved against wherever the user launched
/// from, so a copy of the app run from the Desktop would try to create its
/// cache next to the Desktop (and silently fail if that were read-only), while
/// the same copy run from a terminal wrote somewhere else entirely. Each build
/// now keeps its data beside its own binary.
pub fn default_cache_dir() -> std::path::PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    base.join("data")
}

/// Full path of the OHLCV cache database.
pub fn default_cache_path() -> std::path::PathBuf {
    default_cache_dir().join("cache.db")
}

impl DataService {
    /// Create a new data service with default cache path.
    pub fn new() -> Result<Self> {
        std::fs::create_dir_all(default_cache_dir())?;
        Ok(Self {
            yahoo: YahooProvider::new(),
            coinbase: CoinbaseProvider::new(),
            bhavcopy: BhavcopyProvider::new()?,
            cache: Cache::new(&default_cache_path())?,
        })
    }

    /// Create with custom cache path.
    pub fn with_cache_path<P: AsRef<std::path::Path>>(cache_path: P) -> Result<Self> {
        Ok(Self {
            yahoo: YahooProvider::new(),
            coinbase: CoinbaseProvider::new(),
            bhavcopy: BhavcopyProvider::new()?,
            cache: Cache::new(cache_path)?,
        })
    }

    /// Fetch OHLCV data with caching.
    /// Tries cache first, falls back to provider, updates cache on success.
    pub async fn fetch_ohlcv(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<OhlcvSeries> {
        let interval_str = interval.as_str();

        // Try cache first, restricted to the requested window. The cache is
        // keyed by symbol and interval, so a range check is required: without
        // it a long-range entry (1Y) would also answer a short-range request
        // (1D) and the chart would render far more bars than were asked for.
        let start_ts = Some(start.timestamp());
        let end_ts = Some(end.timestamp());
        if let Ok(Some(candles)) =
            self.cache
                .get_ohlcv_in_range(symbol, interval_str, start_ts, end_ts)
        {
            // The cache is only usable when it actually *covers* the request.
            // A 6-month entry satisfies a 1-year query's range filter while
            // holding only half the bars, which made 1Y and 6M render the
            // identical series under different labels. Require the cached
            // span to reach the requested start (allowing a few gaps for
            // weekends and holidays) before short-circuiting to it.
            if Self::cache_covers(&candles, start.timestamp(), end.timestamp()) {
                let series = OhlcvSeries::new(symbol, candles);
                if series.validate().is_ok() {
                    return Ok(series);
                }
            }
        }

        // Determine provider based on symbol
        let series = if symbol.ends_with("-USD") || symbol.contains("BTC") || symbol.contains("ETH")
        {
            // Use Coinbase for crypto
            let granularity = self.interval_to_granularity(interval);
            let start_ts = Some(start);
            let end_ts = Some(end);
            self.coinbase
                .fetch_candles(symbol, granularity, start_ts, end_ts)
                .await?
        } else {
            // Yahoo is the primary source for everything else. If it fails or
            // returns nothing usable, fall back to the exchange's own daily
            // settlement file for Indian equities, which needs no API key.
            match self.yahoo.fetch_ohlcv(symbol, interval, start, end).await {
                Ok(s) if !s.candles.is_empty() => s,
                Ok(_) | Err(_) => {
                    // The settlement file carries one end-of-day bar per
                    // symbol, so it can only answer daily and coarser
                    // requests. Sub-daily ranges must fail loudly rather than
                    // be silently served a single daily candle.
                    let fallback = if matches!(interval, Interval::Day1 | Interval::Week1) {
                        self.bhavcopy_history(symbol, &start, &end).await?
                    } else {
                        None
                    };
                    fallback.ok_or_else(|| BtError::EmptySeries(symbol.to_string()))?
                }
            }
        };

        // Update cache
        let _ = self.cache.put_ohlcv(symbol, interval_str, &series.candles);

        Ok(series)
    }

    /// True when `candles` spans enough of `[start, end]` to answer the request.
    ///
    /// Daily and weekly series have gaps for weekends, holidays and halts, so
    /// the oldest bar is allowed to sit a few intervals inside the window
    /// before the cache is treated as incomplete.
    fn cache_covers(candles: &[bt_core::Candle], start: i64, end: i64) -> bool {
        if candles.is_empty() {
            return false;
        }
        let first = candles.first().map(|c| c.t as i64).unwrap_or(i64::MAX);
        let last = candles.last().map(|c| c.t as i64).unwrap_or(i64::MIN);
        // Require the series to begin near the request start and to run up to
        // (or very close to) the request end.
        const TOLERANCE_SECS: i64 = 7 * 86_400;
        first <= start + TOLERANCE_SECS && last >= end - TOLERANCE_SECS
    }

    /// Builds history from official NSE daily settlement files.
    ///
    /// Walks back over every day in `[start, end]` collecting the scrip's
    /// settlement bar, so a Yahoo outage still yields a real series rather
    /// than synthetic filler. Returns `None` for non-Indian symbols.
    ///
    /// Only meaningful for daily and coarser intervals: a settlement file has
    /// one end-of-day bar per scrip and cannot answer an intraday request.
    async fn bhavcopy_history(
        &self,
        symbol: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Result<Option<OhlcvSeries>> {
        if !symbol.to_ascii_uppercase().ends_with(".NS") {
            return Ok(None);
        }
        let want = symbol
            .split('.')
            .next()
            .unwrap_or(symbol)
            .to_ascii_uppercase();
        let mut date = end.date_naive();
        let start_date = start.date_naive();
        let mut collected: Vec<bt_core::Candle> = Vec::new();
        let mut attempts = 0;
        // Every weekday in the window is visited, not just the first day that
        // happens to contain the scrip: a one-bar fallback charts as a flat
        // line, which is worse than reporting the outage. Holidays simply 404,
        // so the cap bounds holidays rather than real trading days.
        while date >= start_date && attempts < 400 {
            if let Ok(rows) = self.bhavcopy.fetch_nse_day(date).await {
                for (scrip, c) in rows {
                    // Only keep the requested scrip; the file holds every
                    // listed equity, so without this every symbol would chart
                    // the whole market.
                    if scrip == want {
                        collected.push(c);
                    }
                }
            }
            date -= chrono::Duration::days(1);
            attempts += 1;
        }
        if collected.is_empty() {
            return Ok(None);
        }
        collected.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        collected.dedup_by(|a, b| (a.t - b.t).abs() < f64::EPSILON);
        Ok(Some(OhlcvSeries::new(symbol, collected)))
    }

    /// Fetch real-time quote.
    pub async fn fetch_quote(&self, symbol: &str) -> Result<Quote> {
        if symbol.ends_with("-USD") {
            self.coinbase.fetch_ticker(symbol).await
        } else {
            self.yahoo.fetch_quote(symbol).await
        }
    }

    /// Search symbols.
    pub async fn search_symbols(&self, query: &str) -> Result<Vec<SymbolInfo>> {
        self.yahoo.search_symbols(query).await
    }

    /// Fetch company profile.
    pub async fn fetch_company_profile(&self, symbol: &str) -> Result<CompanyProfile> {
        self.yahoo.fetch_company_profile(symbol).await
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> Result<crate::cache::CacheStats> {
        self.cache.stats()
    }

    /// Cleanup expired cache entries.
    pub fn cleanup_cache(&self) -> Result<usize> {
        self.cache.cleanup()
    }

    fn interval_to_granularity(&self, interval: Interval) -> u32 {
        match interval {
            Interval::Min1 => 60,
            Interval::Min5 => 300,
            Interval::Min15 => 900,
            Interval::Min30 => 1800,
            Interval::Hour1 => 3600,
            Interval::Day1 => 86400,
            Interval::Week1 => 604800,
            Interval::Month1 => 2592000,
        }
    }
}

impl Default for DataService {
    fn default() -> Self {
        Self::new().expect("Failed to create DataService")
    }
}

// Re-exports for convenience
pub use crate::coinbase::CoinbaseProvider as Coinbase;
pub use crate::provider::DataProvider as Provider;
pub use crate::yahoo::YahooProvider as Yahoo;

// India market data re-exports
pub use crate::india::{
    BankingIndicator, CommodityData, CorporateAction, CreditRating, FPIFlow, GSecData, IPOData,
    MacroIndicator, MoneyMarketRate, MutualFundData, OptionsChainEntry, RBIPolicy, USDRate,
    YieldCurvePoint,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_constants() {
        assert!(!COMPANY_LIST.is_empty());
        assert!(COMPANY_LIST.len() >= 50);
    }

    #[test]
    fn test_data_service_creation() {
        let service = DataService::new();
        assert!(service.is_ok());
    }

    fn candle_at(ts: i64) -> bt_core::Candle {
        bt_core::Candle::new(ts as f64, 100.0, 105.0, 99.0, 103.0, 1000.0)
    }

    /// Regression test: a 6-month cache entry satisfied a 1-year request's range
    /// filter while holding only half the bars, so 1Y and 6M rendered the
    /// identical series under different labels.
    #[test]
    fn test_cache_covers_rejects_a_partial_window() {
        let day = 86_400_i64;
        let end = 1_800_000_000_i64;
        // Six months of daily bars.
        let six_months: Vec<bt_core::Candle> =
            (0..180).map(|i| candle_at(end - (180 - i) * day)).collect();

        // A 1-year request must not be satisfied by that.
        let one_year_start = end - 365 * day;
        assert!(
            !DataService::cache_covers(&six_months, one_year_start, end),
            "half the window must not pass as full coverage"
        );
        // But the 6-month request it does cover should pass.
        let six_month_start = end - 180 * day;
        assert!(DataService::cache_covers(&six_months, six_month_start, end));
    }

    #[test]
    fn test_cache_covers_accepts_a_full_window() {
        let day = 86_400_i64;
        let end = 1_800_000_000_i64;
        let year: Vec<bt_core::Candle> =
            (0..365).map(|i| candle_at(end - (365 - i) * day)).collect();
        // A 6-month request against a year of cached bars is covered.
        assert!(DataService::cache_covers(&year, end - 180 * day, end));
        // As is the full year.
        assert!(DataService::cache_covers(&year, end - 365 * day, end));
    }

    /// Weekends and holidays leave small gaps, so a cache whose first bar is a
    /// few days inside the window must still be accepted.
    #[test]
    fn test_cache_covers_tolerates_weekends_and_holidays() {
        let day = 86_400_i64;
        let end = 1_800_000_000_i64;
        // Starts three days late, like a Monday after a Friday request.
        let bars: Vec<bt_core::Candle> = (0..90)
            .map(|i| candle_at(end - (90 - i) * day - 3 * day))
            .collect();
        assert!(DataService::cache_covers(&bars, end - 90 * day, end));
    }

    #[test]
    fn test_cache_covers_rejects_empty_and_stale() {
        let day = 86_400_i64;
        assert!(!DataService::cache_covers(&[], 0, 100));
        // A single bar from long before the window does not cover it.
        let stale = vec![candle_at(1_000 * day)];
        assert!(!DataService::cache_covers(&stale, 1_800 * day, 1_900 * day));
    }
}

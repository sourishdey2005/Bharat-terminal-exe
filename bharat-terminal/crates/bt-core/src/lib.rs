//! bt-core: shared types for Bharat Terminal.
//! Made by Sourish Dey.
//!
//! This crate defines the fundamental data structures (OHLCV candles,
//! time series, error types) used across the visualization layer, plus
//! deterministic synthetic-data generators so every visualization can be
//! rendered and tested locally without any network access or API keys.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;

pub const AUTHOR: &str = "Sourish Dey";
pub const APP_NAME: &str = "BHARAT TERMINAL";
pub const TAGLINE: &str = "Bloomberg power. Zero cost. Made in India.";

/// Errors that can occur anywhere in Bharat Terminal.
#[derive(thiserror::Error, Debug)]
pub enum BtError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("empty series: {0}")]
    EmptySeries(String),
    #[error("render error: {0}")]
    Render(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("database error: {0}")]
    Database(String),
    #[error("data fetch error: {0}")]
    DataFetch(String),
}

pub type Result<T> = std::result::Result<T, BtError>;

/// A single OHLCV candle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candle {
    /// Bar index (or minutes/days since start) — used as the x-axis.
    pub t: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    pub fn new(t: f64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            t,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    pub fn is_bullish(&self) -> bool {
        self.close >= self.open
    }
}

impl fmt::Display for Candle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "t={:.0} O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
            self.t, self.open, self.high, self.low, self.close, self.volume
        )
    }
}

/// A ticker's OHLCV series plus metadata.
#[derive(Debug, Clone)]
pub struct OhlcvSeries {
    pub symbol: String,
    pub candles: Vec<Candle>,
}

impl OhlcvSeries {
    pub fn new(symbol: impl Into<String>, candles: Vec<Candle>) -> Self {
        Self {
            symbol: symbol.into(),
            candles,
        }
    }

    pub fn closes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    pub fn returns(&self) -> Vec<f64> {
        self.candles
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.candles.is_empty() {
            return Err(BtError::EmptySeries(self.symbol.clone()));
        }
        Ok(())
    }
}

/// Deterministic, seeded geometric random walk — stands in for a live feed
/// so every visualization can be demoed and unit-tested offline.
/// Uses daily timestamps starting from 2024-01-01 for realistic x-axis display.
pub fn synthetic_ohlcv(symbol: &str, n: usize, seed: u64, start_price: f64) -> OhlcvSeries {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut candles = Vec::with_capacity(n);
    let mut price = start_price.max(1.0);

    let start_ts = 1704067200_i64;
    let day_secs = 86400_i64;

    for i in 0..n {
        let drift = 0.0002;
        let vol = 0.012;
        let shock: f64 = rng.gen_range(-1.0..1.0);
        let ret = drift + vol * shock;
        let open = price;
        let close = (open * (1.0 + ret)).max(0.01);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(1_000.0..50_000.0);
        let t = (start_ts + i as i64 * day_secs) as f64;
        candles.push(Candle::new(t, open, high, low, close, volume));
        price = close;
    }

    OhlcvSeries::new(symbol, candles)
}

/// Generates a correlated set of return series (for correlation heatmaps,
/// efficient frontier, etc.) using a simple common-factor model.
pub fn synthetic_correlated_returns(
    symbols: &[&str],
    n: usize,
    seed: u64,
) -> Vec<(String, Vec<f64>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let factor: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0) * 0.01).collect();

    symbols
        .iter()
        .enumerate()
        .map(|(idx, sym)| {
            let beta = 0.3 + 0.15 * idx as f64;
            let series: Vec<f64> = (0..n)
                .map(|i| beta * factor[i] + rng.gen_range(-1.0..1.0) * 0.006)
                .collect();
            (sym.to_string(), series)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_series_has_requested_length() {
        let s = synthetic_ohlcv("RELIANCE.NS", 100, 42, 2500.0);
        assert_eq!(s.candles.len(), 100);
        s.validate().unwrap();
    }

    #[test]
    fn synthetic_series_is_deterministic_for_seed() {
        let a = synthetic_ohlcv("AAPL", 50, 7, 100.0);
        let b = synthetic_ohlcv("AAPL", 50, 7, 100.0);
        assert_eq!(a.candles, b.candles);
    }

    #[test]
    fn candle_ohlc_relationship_holds() {
        let s = synthetic_ohlcv("TEST", 200, 1, 50.0);
        for c in &s.candles {
            assert!(c.high >= c.open.max(c.close) - 1e-9);
            assert!(c.low <= c.open.min(c.close) + 1e-9);
        }
    }

    #[test]
    fn returns_length_is_n_minus_one() {
        let s = synthetic_ohlcv("X", 30, 3, 10.0);
        assert_eq!(s.returns().len(), 29);
    }

    #[test]
    fn empty_series_fails_validation() {
        let s = OhlcvSeries::new("EMPTY", vec![]);
        assert!(s.validate().is_err());
    }

    #[test]
    fn correlated_returns_have_matching_lengths() {
        let data = synthetic_correlated_returns(&["A", "B", "C"], 60, 9);
        assert_eq!(data.len(), 3);
        for (_, series) in &data {
            assert_eq!(series.len(), 60);
        }
    }
}

/// One side of an option strike: exchange-published positioning and price.
///
/// Shared by the NSE and US (Yahoo) providers in bt-data and the math in
/// bt-analytics, so both sides agree on field meaning without either crate
/// depending on the other.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OptionLeg {
    /// Open interest in contracts.
    pub oi: u64,
    /// Day change in OI (NSE publishes it; Yahoo legs leave it 0).
    pub oi_change: i64,
    /// Traded volume in contracts.
    pub volume: u64,
    /// Last traded price.
    pub ltp: f64,
    /// Implied volatility as a decimal (0.22, not 22).
    pub iv: f64,
}

/// One strike row with both legs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OptionStrike {
    pub strike: f64,
    pub call: OptionLeg,
    pub put: OptionLeg,
    /// Nearest strike to the underlying at fetch time.
    pub atm: bool,
}

/// A full chain for one symbol + expiry, in exchange-neutral shape.
#[derive(Debug, Clone)]
pub struct OptionChain {
    pub symbol: String,
    /// Expiry as published (`"30-Oct-2026"` for NSE, unix seconds as text
    /// for Yahoo) — opaque here, parsed by whoever needs it.
    pub expiry: String,
    pub underlying_value: f64,
    pub strikes: Vec<OptionStrike>,
    /// Fetch time, seconds since epoch, for staleness display.
    pub fetched_at: u64,
}

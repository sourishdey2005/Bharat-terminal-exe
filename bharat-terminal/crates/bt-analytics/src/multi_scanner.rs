// crates/bt-analytics/src/multi_scanner.rs
// Author: Sourish Dey

//! Runs the advisor rules across a whole watchlist and ranks the results.
//!
//! This is what makes the Advisor useful beyond one chart: the same
//! deterministic rules from [`crate::advisor`] applied to every symbol, ordered
//! by how strongly the inputs agree. It computes no new signals of its own, so
//! a symbol's rank here can always be traced back to the individual readings that
//! produced it.
//!
//! Symbols with no measured data are excluded rather than ranked last. Putting a
//! symbol that never loaded at the bottom would make "no data" look like "the
//! weakest setup", which is the opposite of true.

use crate::advisor::{Action, AdvisorInput, advise};
use bt_core::Candle;
use std::collections::HashMap;

/// One symbol's result.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanRow {
    pub symbol: String,
    pub last_price: f64,
    pub action: Action,
    pub confidence: f64,
    /// Net rule score. Higher is more bullish.
    pub score: i32,
    pub inputs_seen: usize,
    /// The strongest supporting or opposing reason, for the one-line summary.
    pub headline: String,
}

impl ScanRow {
    /// Sort key: score first, then how many inputs backed it.
    ///
    /// Ties are broken by `inputs_seen` deliberately. A +2 built from three
    /// agreeing indicators is a stronger observation than a +2 from one, and
    /// without this the ranking would treat them as identical.
    pub fn rank_key(&self) -> (i32, usize) {
        (self.score, self.inputs_seen)
    }
}

/// Build an [`AdvisorInput`] for `symbol` from its candles.
///
/// This is the bridge between raw candles and the rule engine: it computes
/// exactly the indicators the rules consume, and nothing else. Indicators that
/// cannot be computed from the available history stay `None` rather than being
/// defaulted, so a short history yields fewer inputs and a lower-confidence row
/// instead of a confident one built on noise.
pub fn advisor_input_for(symbol: &str, candles: &[Candle]) -> AdvisorInput {
    let series = bt_core::OhlcvSeries::new(symbol, candles.to_vec());
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let last_price = closes.last().copied().unwrap_or(0.0);

    // Every indicator below returns a full-length series padded with NaN until
    // it has enough history, so the last *finite* value is the real reading.
    let rsi = last_finite(&crate::indicators::rsi(&series, 14));

    // MACD minus its signal line is the convention the advisor expects.
    let (macd_line, signal_line, _) = crate::indicators::macd(&series);
    let macd_signal = match (last_finite(&macd_line), last_finite(&signal_line)) {
        (Some(m), Some(s)) => Some(m - s),
        _ => None,
    };

    let price_vs = |period: usize| -> Option<f64> {
        if closes.len() < period {
            return None;
        }
        let sma = closes[closes.len() - period..].iter().sum::<f64>() / period as f64;
        if sma.abs() > 1e-12 {
            Some((last_price - sma) / sma * 100.0)
        } else {
            None
        }
    };

    let atr_pct = last_finite(&crate::indicators::atr(&series, 14))
        .filter(|a| *a > 0.0)
        .filter(|_| last_price.abs() > 1e-12)
        .map(|a| a / last_price * 100.0);

    let volume_ratio = if candles.len() < 21 {
        None
    } else {
        let recent = candles[candles.len() - 1].volume;
        let base = candles[candles.len() - 21..candles.len() - 1]
            .iter()
            .map(|c| c.volume)
            .sum::<f64>()
            / 20.0;
        (base > 1e-12 && recent.is_finite()).then(|| recent / base)
    };

    AdvisorInput {
        symbol: symbol.to_string(),
        last_price,
        rsi,
        macd_signal,
        // The scanner has no forecast input: running one forecaster per symbol
        // is far too slow for a watchlist sweep. The Forecast column stays
        // empty rather than being filled with a cheap substitute that would
        // look like the same signal.
        forecast_change_pct: None,
        watchsignal: None,
        price_vs_sma20: price_vs(20),
        price_vs_sma50: price_vs(50),
        atr_pct,
        volume_ratio,
    }
}

/// Apply the rules to every symbol and return the rows ranked strongest-first.
///
/// `series` maps symbol to candles. A symbol with too little history to produce
/// any indicator is skipped entirely.
pub fn scan(series: &HashMap<String, Vec<Candle>>) -> Vec<ScanRow> {
    let mut rows: Vec<ScanRow> = series
        .iter()
        .filter(|(_, c)| c.len() >= 15)
        .map(|(symbol, candles)| {
            let input = advisor_input_for(symbol, candles);
            let out = advise(&input);
            ScanRow {
                symbol: symbol.clone(),
                last_price: input.last_price,
                action: out.action,
                confidence: out.confidence,
                score: out.score,
                inputs_seen: out.inputs_seen,
                headline: out.reasons.first().cloned().unwrap_or_default(),
            }
        })
        .filter(|r| r.inputs_seen > 0)
        .collect();

    rows.sort_by(|a, b| {
        b.rank_key()
            .cmp(&a.rank_key())
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    rows
}

/// Symbols the rules read as bullish, strongest first.
pub fn bullish(rows: &[ScanRow]) -> Vec<&ScanRow> {
    rows.iter()
        .filter(|r| matches!(r.action, Action::Buy | Action::StrongBuy))
        .collect()
}

/// Symbols the rules read as bearish, strongest first.
pub fn bearish(rows: &[ScanRow]) -> Vec<&ScanRow> {
    rows.iter()
        .filter(|r| matches!(r.action, Action::Sell | Action::StrongSell))
        .collect()
}

/// Fraction of rows that resolved to something other than `Wait`.
///
/// Returns `None` for an empty scan: "0% actionable" and "nothing to scan" are
/// different answers.
pub fn actionable_fraction(rows: &[ScanRow]) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    let n = rows.iter().filter(|r| r.action != Action::Wait).count();
    Some(n as f64 / rows.len() as f64)
}

/// Last finite value in a series.
fn last_finite(xs: &[f64]) -> Option<f64> {
    xs.iter().rev().copied().find(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rising series: alternating down/up days with a small net upward drift.
    ///
    /// Deliberately *not* a straight ramp. A ramp makes a new high on every bar,
    /// which pins RSI at 100, and the advisor then reports the most bullish-looking
    /// series as overbought — correct behaviour for RSI, useless as a fixture.
    /// Alternating days hold RSI near the neutral band so the trend inputs carry
    /// the verdict, which is the thing under test.
    fn uptrend(n: usize) -> Vec<Candle> {
        drift(n, 0.25, -0.8, 0.9)
    }

    /// The mirror image: alternating days with a small net downward drift.
    fn downtrend(n: usize) -> Vec<Candle> {
        drift(n, -0.25, 0.9, -0.8)
    }

    /// Build a continuous series from a per-day step pattern.
    ///
    /// `up_step` applies on odd bars, `down_step` on even ones; `drift` is added
    /// to every bar's close so the series trends. Opens are set to the prior
    /// close, which keeps the series continuous so no artificial gaps appear.
    fn drift(n: usize, drift: f64, down_step: f64, up_step: f64) -> Vec<Candle> {
        let mut close = 100.0;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let open = close;
            let step = if i % 2 == 0 { down_step } else { up_step };
            close += drift + step;
            out.push(Candle::new(
                i as f64,
                open,
                close.max(open) + 0.3,
                close.min(open) - 0.3,
                close,
                1000.0,
            ));
        }
        out
    }

    fn basket(entries: Vec<(&str, Vec<Candle>)>) -> HashMap<String, Vec<Candle>> {
        entries
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    #[test]
    fn an_uptrend_scores_above_a_downtrend() {
        let rows = scan(&basket(vec![
            ("UP", uptrend(80)),
            ("DOWN", downtrend(80)),
        ]));
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!(rows[0].symbol, "UP", "{rows:?}");
        assert!(rows[0].score > rows[1].score, "{rows:?}");
    }

    #[test]
    fn ranking_is_deterministic_for_identical_symbols() {
        // Two identical uptrends must order by symbol, not by hash order.
        let rows = scan(&basket(vec![
            ("ZZZ", uptrend(80)),
            ("AAA", uptrend(80)),
        ]));
        assert_eq!(rows[0].symbol, "AAA", "{rows:?}");
        assert_eq!(scan(&basket(vec![
            ("ZZZ", uptrend(80)),
            ("AAA", uptrend(80)),
        ]))[0].symbol, "AAA");
    }

    #[test]
    fn symbols_with_too_little_history_are_excluded_not_ranked_last() {
        // Excluded, because "no data" must not masquerade as "weakest setup".
        let rows = scan(&basket(vec![
            ("GOOD", uptrend(80)),
            ("STUB", uptrend(5)),
        ]));
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].symbol, "GOOD");
        assert!(!rows.iter().any(|r| r.symbol == "STUB"));
    }

    #[test]
    fn an_empty_basket_scans_to_nothing() {
        assert!(scan(&basket(vec![])).is_empty());
    }

    #[test]
    fn every_row_carries_a_price_and_at_least_one_input() {
        let rows = scan(&basket(vec![
            ("A", uptrend(80)),
            ("B", downtrend(80)),
            ("C", uptrend(60)),
        ]));
        for r in &rows {
            assert!(r.last_price > 0.0, "{r:?}");
            assert!(r.inputs_seen > 0, "{r:?}");
            assert!(r.confidence > 0.0, "{r:?}");
            assert!(r.confidence <= 1.0, "{r:?}");
        }
    }

    #[test]
    fn uptrend_is_bullish_and_downtrend_is_not() {
        let rows = scan(&basket(vec![
            ("UP", uptrend(120)),
            ("DOWN", downtrend(120)),
        ]));
        let up = rows.iter().find(|r| r.symbol == "UP").unwrap();
        let down = rows.iter().find(|r| r.symbol == "DOWN").unwrap();
        // The bull claim is about the uptrend being more bullish than the
        // downtrend. A mild downtrend legitimately reads as Reduce rather than
        // Sell, so only the bullish side is asserted to a specific action.
        assert!(
            matches!(up.action, Action::Buy | Action::StrongBuy),
            "a rising series must read bullish, got {:?} (score {})",
            up.action,
            up.score
        );
        assert!(
            !matches!(down.action, Action::Buy | Action::StrongBuy),
            "a falling series must not read bullish: {down:?}"
        );
        assert!(up.score > down.score, "{rows:?}");
        assert!(bullish(&rows).iter().any(|r| r.symbol == "UP"));
    }

    #[test]
    fn a_short_history_yields_fewer_inputs_rather_than_confident_ones() {
        // 20 bars is enough for RSI but not for SMA50, so the rule engine must
        // see fewer inputs on a short history than on a long one.
        let short = advisor_input_for("X", &uptrend(20));
        let long = advisor_input_for("X", &uptrend(120));
        assert!(short.price_vs_sma50.is_none());
        assert!(long.price_vs_sma50.is_some());
        assert!(
            advise(&short).inputs_seen < advise(&long).inputs_seen,
            "short {:?} vs long {:?}",
            advise(&short).inputs_seen,
            advise(&long).inputs_seen
        );
    }

    #[test]
    fn no_row_contains_a_nan_or_infinity() {
        let rows = scan(&basket(vec![
            ("A", uptrend(80)),
            ("B", downtrend(80)),
            ("C", uptrend(20)),
        ]));
        for r in &rows {
            assert!(r.last_price.is_finite(), "{r:?}");
            assert!(r.confidence.is_finite(), "{r:?}");
            assert!(!r.headline.contains("NaN"), "{r:?}");
        }
    }

    #[test]
    fn a_constant_series_does_not_produce_nan_inputs() {
        let flat: Vec<Candle> = (0..80)
            .map(|i| Candle::new(i as f64, 100.0, 100.0, 100.0, 100.0, 1000.0))
            .collect();
        let input = advisor_input_for("FLAT", &flat);
        assert_eq!(input.last_price, 100.0);
        let out = advise(&input);
        assert!(out.confidence.is_finite());
        assert!((-10..=10).contains(&out.score), "{out:?}");
    }

    #[test]
    fn actionable_fraction_is_none_for_an_empty_scan() {
        assert_eq!(actionable_fraction(&[]), None);
        let rows = scan(&basket(vec![("A", uptrend(80))]));
        let f = actionable_fraction(&rows).expect("a fraction");
        assert!((0.0..=1.0).contains(&f), "{f}");
    }

    #[test]
    fn ties_break_on_input_count_then_symbol() {
        // Construct two rows with equal scores but different evidence counts.
        let a = ScanRow {
            symbol: "A".into(),
            last_price: 1.0,
            action: Action::Buy,
            confidence: 0.5,
            score: 2,
            inputs_seen: 3,
            headline: String::new(),
        };
        let b = ScanRow {
            inputs_seen: 5,
            ..a.clone()
        };
        assert!(b.rank_key() > a.rank_key(), "more evidence must rank higher");
    }
}
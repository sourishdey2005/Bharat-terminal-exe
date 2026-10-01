// crates/bt-analytics/src/backtest.rs
// Author: Sourish Dey

//! Strategy backtesting over close series.
//!
//! Pure functions: a strategy maps history to a position series, and the
//! engine turns positions into an equity curve with Sharpe and max drawdown.
//! No data fetching, no models, no I/O — the CLI and (later) the UI supply the
//! closes and render the curve.

/// One backtest run.
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestResult {
    pub strategy: String,
    /// Equity per bar, starting at 1.0.
    pub equity: Vec<f64>,
    /// Total return as a fraction (0.12 = +12%).
    pub total_return: f64,
    /// Annualised Sharpe, assuming 252 trading days and zero risk-free rate.
    pub sharpe: f64,
    /// Worst peak-to-trough fall as a fraction.
    pub max_drawdown: f64,
    /// Buy-and-hold total return over the same window, for comparison.
    pub buy_hold_return: f64,
    /// Fraction of bars spent in the market.
    pub exposure: f64,
}

/// Position series: 1.0 long, 0.0 flat. Shorting is out of scope for the
/// cash-equity strategies this engine targets.
pub type PositionSeries = Vec<f64>;

/// SMA-cross positions: long when the fast average is above the slow one.
///
/// `fast` and `slow` are window lengths in bars. Needs at least `slow` bars;
/// shorter input yields an all-flat series rather than an error, so a thin
/// history backtests as "never traded" instead of crashing the caller.
pub fn sma_cross_positions(closes: &[f64], fast: usize, slow: usize) -> PositionSeries {
    let n = closes.len();
    let mut pos = vec![0.0; n];
    if n < slow.max(2) || fast == 0 || slow == 0 || fast >= slow {
        return pos;
    }
    let sma = |i: usize, w: usize| -> f64 {
        closes[i + 1 - w..=i].iter().sum::<f64>() / w as f64
    };
    for (i, p) in pos.iter_mut().enumerate().skip(slow - 1) {
        *p = if sma(i, fast) > sma(i, slow) { 1.0 } else { 0.0 };
    }
    pos
}

/// Run positions against closes.
pub fn run_backtest(strategy: &str, closes: &[f64], positions: &[f64]) -> Option<BacktestResult> {
    if closes.len() < 2 || positions.len() != closes.len() {
        return None;
    }
    let mut equity = Vec::with_capacity(closes.len());
    let mut eq = 1.0;
    equity.push(eq);
    let mut in_market = 0usize;
    for i in 1..closes.len() {
        let prev = closes[i - 1];
        let ret = if prev.abs() > 1e-12 {
            closes[i] / prev - 1.0
        } else {
            0.0
        };
        // Position decided at bar i-1 applies to the i-1 → i move (no lookahead).
        let p = positions[i - 1].clamp(0.0, 1.0);
        eq *= 1.0 + p * ret;
        if !eq.is_finite() {
            return None;
        }
        equity.push(eq);
        if p > 0.0 {
            in_market += 1;
        }
    }
    let total_return = eq - 1.0;
    let base = closes[0].abs().max(1e-12);
    let buy_hold_return = closes[closes.len() - 1] / base - 1.0;

    let mut rets: Vec<f64> = equity.windows(2).map(|w| w[1] / w[0] - 1.0).collect();
    rets.retain(|r| r.is_finite());
    let sharpe = if rets.len() > 1 {
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let var =
            rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
        if var > 0.0 {
            mean / var.sqrt() * (252.0_f64).sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut peak: f64 = f64::MIN;
    let mut max_dd: f64 = 0.0;
    for &e in &equity {
        peak = peak.max(e);
        if peak > 0.0 {
            max_dd = max_dd.max((peak - e) / peak);
        }
    }

    Some(BacktestResult {
        strategy: strategy.to_string(),
        equity,
        total_return,
        sharpe,
        max_drawdown: max_dd,
        buy_hold_return,
        exposure: in_market as f64 / (closes.len() - 1).max(1) as f64,
    })
}

/// Convenience: SMA-cross backtest in one call.
pub fn backtest_sma_cross(closes: &[f64], fast: usize, slow: usize) -> Option<BacktestResult> {
    let pos = sma_cross_positions(closes, fast, slow);
    run_backtest(&format!("sma-cross({fast},{slow})"), closes, &pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rising(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64).collect()
    }

    #[test]
    fn rising_market_stays_long_after_warmup() {
        let closes = rising(100);
        let pos = sma_cross_positions(&closes, 10, 30);
        // Index 29 is the first bar with a full 30-bar window; before it,
        // no average exists so the series stays flat.
        assert!(pos[..29].iter().all(|&p| p == 0.0));
        assert!(pos[29..].iter().all(|&p| p == 1.0));
        let r = run_backtest("t", &closes, &pos).unwrap();
        // Warmup bars cannot trade, so total trails buy-and-hold by exactly
        // the missed early moves — but stays well positive on a ramp.
        assert!(r.total_return > 0.5 * r.buy_hold_return);
        assert!((r.exposure - 70.0 / 99.0).abs() < 1e-9);
        assert!(r.sharpe > 2.0, "smooth ramp should Sharpe well, got {}", r.sharpe);
        assert_eq!(r.max_drawdown, 0.0);
    }

    #[test]
    fn choppy_market_trades_less_and_loses_less_than_hold() {
        // Sawtooth: buy-and-hold ends flat, cross strategy must not blow up.
        let closes: Vec<f64> = (0..120)
            .map(|i| 100.0 + 5.0 * ((i as f64 * 0.5).sin()))
            .collect();
        let r = backtest_sma_cross(&closes, 5, 20).unwrap();
        assert!(r.equity.iter().all(|e| e.is_finite()));
        assert!(r.max_drawdown >= 0.0 && r.max_drawdown < 1.0);
        assert!((0.0..=1.0).contains(&r.exposure));
    }

    #[test]
    fn invalid_windows_trade_flat() {
        let closes = rising(50);
        assert!(sma_cross_positions(&closes, 30, 10).iter().all(|&p| p == 0.0));
        assert!(sma_cross_positions(&closes, 0, 10).iter().all(|&p| p == 0.0));
        assert!(sma_cross_positions(&closes[..10], 5, 20).iter().all(|&p| p == 0.0));
    }

    #[test]
    fn degenerate_inputs_return_none_not_panic() {
        assert!(run_backtest("t", &[], &[]).is_none());
        assert!(run_backtest("t", &[100.0], &[1.0]).is_none());
        assert!(run_backtest("t", &[100.0, 101.0], &[1.0]).is_none());
        assert!(run_backtest("t", &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]).is_some());
    }

    #[test]
    fn drawdown_measures_the_worst_fall() {
        // Up 10%, then down 20%, then flat: max DD is 20% of the peak.
        let closes = vec![100.0, 110.0, 88.0, 88.0, 88.0];
        let r = run_backtest("t", &closes, &[1.0; 5]).unwrap();
        assert!((r.max_drawdown - 0.2).abs() < 1e-9, "dd={}", r.max_drawdown);
    }
}

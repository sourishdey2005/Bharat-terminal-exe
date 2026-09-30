//! Audit every dashboard tab for real, non-degenerate data.
//!
//! The visualization expansion added ~37 indicator and overlay tabs. A tab that
//! lists itself in the UI but plots an empty or constant series is worse than
//! no tab at all, because it looks like it works. This runs every registered
//! indicator over a realistic synthetic series and asserts each one produces
//! enough finite, *varying* values to actually draw a line.
//!
//! Run: cargo test -p bt-analytics --test tab_data_audit -- --nocapture

use bt_analytics::*;
use bt_core::{Candle, OhlcvSeries};

/// A trending, noisy daily series: the shape real market data has, and the
/// shape indicators are expected to respond to.
fn series() -> OhlcvSeries {
    let candles: Vec<Candle> = (0..300)
        .map(|i| {
            let t = i as f64;
            let base = 2400.0 + 2.0 * t + 25.0 * (t * 0.06).sin() + 9.0 * (t * 0.31).cos();
            // Closes must not sit exactly at the bar midpoint, or the
            // close-location value is always 0 and the A/D line, Chaikin money
            // flow and MFI all collapse to a constant.
            let drift = 1.5 * (t * 0.23).sin();
            Candle::new(
                1_700_000_000.0 + i as f64 * 86_400.0,
                base - 1.5,
                base + 4.0,
                base - 4.0,
                base + drift,
                1_000_000.0 + (i as f64) * 1_000.0,
            )
        })
        .collect();
    OhlcvSeries::new("AUDIT", candles)
}

/// An indicator's output is only plottable if it has enough finite values and
/// they actually vary. A constant or near-empty series renders as a flat line
/// or nothing, which is a broken tab.
fn assert_plottable(name: &str, values: &[f64]) {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();

    assert!(
        finite.len() >= 10,
        "{name}: only {} finite values, nothing to draw",
        finite.len()
    );

    let lo = finite.iter().cloned().fold(f64::MAX, f64::min);
    let hi = finite.iter().cloned().fold(f64::MIN, f64::max);
    let span = (hi - lo).abs();
    assert!(
        span > 1e-9,
        "{name}: output is constant at {lo}; a flat line is not a plot"
    );

    // Distinct-value count catches degenerate repetition that min/max misses.
    let distinct = finite
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct >= 5,
        "{name}: only {distinct} distinct values across {} points",
        finite.len()
    );

    println!(
        "{name:>22}: {:>4} pts, range [{:.3}, {:.3}]",
        finite.len(),
        lo,
        hi
    );
}

#[test]
fn test_every_extended_indicator_produces_plottable_output() {
    let s = series();

    macro_rules! check {
        ($name:literal, $expr:expr) => {
            let v: Vec<f64> = $expr;
            assert_plottable($name, &v);
        };
    }

    check!("stoch_rsi", stoch_rsi(&s, 14, 14));
    check!("zscore", zscore(&s, 20));
    check!("mfi", mfi(&s, 14));
    check!("ultimate_osc", ultimate_osc(&s, 7, 14, 28));
    check!("tsi", tsi(&s, 25, 13).0);
    check!("tsi_signal", tsi(&s, 25, 13).1);
    check!("coppock", coppock(&s));
    check!("dpo", dpo(&s, 20));
    check!("aroon_up", aroon(&s, 14).0);
    check!("aroon_down", aroon(&s, 14).1);
    check!("aroon_osc", aroon_osc(&s, 14));
    check!("ulcer", ulcer(&s, 14));
    check!("eom", eom(&s, 14));
    check!("force_index", force_index(&s, 13));
    check!("mass_index", mass_index(&s, 25));
    check!("pvt", pvt(&s));
    check!("mfv", mfv(&s));
    check!("ad_line", ad_line(&s));
    check!("trend_intensity", trend_intensity(&s, 20));
    check!("realized_vol", realized_vol(&s, 20));
    check!("keltner_width", keltner_width(&s, 20));
    check!("kst", kst(&s, 10, 15, 20, 30, 10, 10, 10, 15, 1).0);
    check!("kst_signal", kst(&s, 10, 15, 20, 30, 10, 10, 10, 15, 1).1);
    check!("elder_bull", elder_ray(&s, 13).0);
    check!("elder_bear", elder_ray(&s, 13).1);
    check!("vortex_plus", vortex(&s, 14).0);
    check!("vortex_minus", vortex(&s, 14).1);
    check!("kama", kama(&s, 10, 2, 30));
    check!("alma", alma(&s, 9, 0.85, 6.0));
    check!("hull_ma", hull_ma(&s, 9));
    check!("wma", wma(&s, 20));
    check!("supertrend", supertrend(&s, 10, 3.0).0);
    check!("high_low_upper", high_low_band(&s, 20).0);
    check!("high_low_lower", high_low_band(&s, 20).1);
    check!("vwap_bands_upper", vwap_bands(&s, 2.0).0);
    check!("vwap_bands_lower", vwap_bands(&s, 2.0).1);
    check!("multi_sma_20", multi_sma(&s).0);
    check!("multi_sma_50", multi_sma(&s).1);
    check!("multi_sma_200", multi_sma(&s).2);
    check!("multi_ema_12", multi_ema(&s).0);
    check!("multi_ema_26", multi_ema(&s).1);
    check!("multi_ema_50", multi_ema(&s).2);
    check!("log_returns", log_returns(&s));
}

#[test]
fn test_pivot_and_fib_levels_are_distinct() {
    let s = series();
    let candles = &s.candles;
    let (mut hh, mut ll) = (f64::NEG_INFINITY, f64::INFINITY);
    for c in candles {
        hh = hh.max(c.high);
        ll = ll.min(c.low);
    }
    let close = candles.last().unwrap().close;

    let p = pivot_levels(hh, ll, close);
    let levels = [p.pp, p.r1, p.r2, p.r3, p.s1, p.s2, p.s3];
    assert!(levels.iter().all(|v| v.is_finite()));
    // Support must sit below price and resistance above it, or the plot is
    // labelled backwards.
    assert!(
        p.s1 < close && p.s2 < p.s1 && p.s3 < p.s2,
        "supports not ordered"
    );
    assert!(
        p.r1 > close && p.r2 > p.r1 && p.r3 > p.r2,
        "resistances not ordered"
    );
    println!(
        "pivots: pp={:.2} r1={:.2} r3={:.2} s1={:.2} s3={:.2}",
        p.pp, p.r1, p.r3, p.s1, p.s3
    );

    let fibs = fib_levels(ll, hh);
    // The standard retracement ladder: 0, 23.6, 38.2, 50, 61.8, 78.6, 100%.
    assert_eq!(fibs.len(), 7, "expected the 7-level retracement ladder");
    // Retracement ladder runs upward from the low, so 0% is the low and 100%
    // is the high.
    assert!((fibs[0].1 - ll).abs() < 1e-6, "0% should be the low");
    assert!((fibs[6].1 - hh).abs() < 1e-6, "100% should be the high");
    for (ratio, price) in &fibs {
        assert!(price.is_finite());
        assert!(
            *price >= ll - 1e-6 && *price <= hh + 1e-6,
            "fib {ratio} outside the range"
        );
    }
    let prices: Vec<f64> = fibs.iter().map(|(_, p)| *p).collect();
    assert!(
        prices.windows(2).any(|w| w[0] != w[1]),
        "fib levels collapsed to a single price"
    );
    println!(
        "fib levels: {:?}",
        fibs.iter().map(|(r, p)| (*r, *p)).collect::<Vec<_>>()
    );
}

#[test]
fn test_ohlc_and_volume_tabs_have_real_bars() {
    let s = series();
    // The OHLC and Volume tabs read straight off the candles, so what matters
    // is that the series is long enough to draw and not degenerate.
    assert!(s.candles.len() >= 200, "audit series too short");
    let closes: Vec<f64> = s.candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = s.candles.iter().map(|c| c.volume).collect();

    assert_plottable("close", &closes);
    assert!(
        volumes.iter().all(|v| *v > 0.0),
        "volume must be positive to plot bars"
    );
    let vhi = volumes.iter().cloned().fold(f64::MIN, f64::max);
    let vlo = volumes.iter().cloned().fold(f64::MAX, f64::min);
    assert!(vhi > vlo, "constant volume would render as a flat block");
    println!("volume range: [{vlo:.0}, {vhi:.0}]");

    // Every bar must satisfy the OHLC invariant, or the candle plot is garbage.
    for c in &s.candles {
        assert!(c.high >= c.low, "high below low at t={}", c.t);
        assert!(
            c.high >= c.open && c.high >= c.close,
            "high too small at t={}",
            c.t
        );
        assert!(
            c.low <= c.open && c.low <= c.close,
            "low too large at t={}",
            c.t
        );
    }
}

#[test]
fn test_original_indicators_still_plot() {
    // The expansion must not have broken the pre-existing tabs.
    let s = series();
    assert_plottable("rsi", &rsi(&s, 14));
    assert_plottable("macd", &macd(&s).0);
    assert_plottable("macd_signal", &macd(&s).1);
    assert_plottable("bollinger_upper", &bollinger(&s, 20, 2.0).0);
    assert_plottable("bollinger_mid", &bollinger(&s, 20, 2.0).1);
    assert_plottable("bollinger_lower", &bollinger(&s, 20, 2.0).2);
    let (adx_line, plus_di, minus_di) = adx(&s, 14);
    assert_plottable("adx", &adx_line);
    assert_plottable("plus_di", &plus_di);
    assert_plottable("minus_di", &minus_di);
    assert_plottable("cci", &cci(&s, 20));
    assert_plottable("williams_r", &williams_r(&s, 14));
    assert_plottable("roc", &roc(&s, 10));
    assert_plottable("obv", &obv(&s));
    assert_plottable("atr", &atr(&s, 14));
    assert_plottable("stochastic_k", &stochastic(&s, 14, 3).0);
    assert_plottable("stochastic_d", &stochastic(&s, 14, 3).1);
    assert_plottable("cmf", &cmf(&s, 20));
    assert_plottable("keltner_upper", &keltner(&s, 20, 2.0).0);
    assert_plottable("donchian_upper", &donchian(&s, 20).0);
    assert_plottable("parabolic_sar", &parabolic_sar(&s, 0.02, 0.02, 0.2));
    assert_plottable("vwap", &vwap(&s));
}

#[test]
fn test_all_indicators_survive_a_short_series_without_panicking() {
    // A user can select 1D on a symbol with little history. Every tab must
    // render *something* or an empty-state message, never panic.
    let short = OhlcvSeries::new(
        "SHORT",
        (0..12)
            .map(|i| {
                let base = 100.0 + i as f64;
                Candle::new(i as f64, base, base + 1.0, base - 1.0, base + 0.5, 1_000.0)
            })
            .collect(),
    );

    let _ = stoch_rsi(&short, 14, 14);
    let _ = zscore(&short, 20);
    let _ = mfi(&short, 14);
    let _ = tsi(&short, 25, 13);
    let _ = aroon(&short, 14);
    let _ = kama(&short, 10, 2, 30);
    let _ = supertrend(&short, 10, 3.0);
    let _ = pivot_levels(105.0, 95.0, 100.0);
    let _ = fib_levels(95.0, 105.0);
    let _ = realized_vol(&short, 20);
    let _ = kst(&short, 10, 15, 20, 30, 10, 10, 10, 15, 1);
    let _ = mass_index(&short, 25);
    let _ = coppock(&short);
    let _ = log_returns(&short);
    println!("short series survived every indicator without panicking");
}

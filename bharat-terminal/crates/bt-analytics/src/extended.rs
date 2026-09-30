// crates/bt-analytics/src/extended.rs
// Author: Sourish Dey

//! Extended indicator library: momentum oscillators, volume analytics,
//! volatility gauges, adaptive trend overlays and level helpers.
//!
//! Conventions (same as `indicators.rs`):
//! - Every series function returns a full-length `Vec<f64>` with `NaN`
//!   warmup, so callers can index by bar without re-aligning.
//! - Inputs with a zero period, or a series shorter than the warmup, yield
//!   all-`NaN` instead of panicking.

use bt_core::OhlcvSeries;

fn closes(series: &OhlcvSeries) -> Vec<f64> {
    series.candles.iter().map(|c| c.close).collect()
}

fn ema_of(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n < period {
        return out;
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut ema = values[..period].iter().sum::<f64>() / period as f64;
    out[period - 1] = ema;
    for i in period..n {
        ema = values[i] * k + ema * (1.0 - k);
        out[i] = ema;
    }
    out
}

fn sma_of(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n < period {
        return out;
    }
    let mut sum = 0.0;
    for i in 0..n {
        sum += values[i];
        if i >= period {
            sum -= values[i - period];
        }
        if i + 1 >= period {
            out[i] = sum / period as f64;
        }
    }
    out
}

fn stdev_of(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n < period {
        return out;
    }
    for i in (period - 1)..n {
        let w = &values[i + 1 - period..=i];
        let mean = w.iter().sum::<f64>() / period as f64;
        let var = w.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / period as f64;
        out[i] = var.sqrt();
    }
    out
}

/// Stochastic RSI (0-100): Stochastic of RSI over `period` with an RSI lookback
/// of `rsi_period`.
pub fn stoch_rsi(series: &OhlcvSeries, period: usize, rsi_period: usize) -> Vec<f64> {
    use crate::rsi;
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || rsi_period == 0 || n <= period + rsi_period {
        return out;
    }
    let rsi_vals = rsi(series, rsi_period);
    for i in (rsi_period + period)..n {
        let w = &rsi_vals[i + 1 - period..=i];
        if w.iter().any(|v| v.is_nan()) {
            continue;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in w {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        out[i] = if hi > lo {
            // Clamped: float rounding can otherwise print 100.00000000000001
            // when the current RSI equals the window extreme.
            (100.0 * (rsi_vals[i] - lo) / (hi - lo)).clamp(0.0, 100.0)
        } else {
            50.0
        };
    }
    out
}

/// True Strength Index (×100) with its 7-bar signal line.
pub fn tsi(series: &OhlcvSeries, r: usize, s: usize) -> (Vec<f64>, Vec<f64>) {
    let n = series.candles.len();
    let mut tsi_line = vec![f64::NAN; n];
    let mut signal = vec![f64::NAN; n];
    if r == 0 || s == 0 || n <= r + s + 1 {
        return (tsi_line, signal);
    }
    let c = closes(series);
    let mut pc = vec![0.0; n];
    let mut apc = vec![0.0; n];
    for i in 1..n {
        pc[i] = c[i] - c[i - 1];
        apc[i] = pc[i].abs();
    }
    let e1 = ema_of(&pc, r);
    let e2 = ema_of(
        &e1.iter()
            .map(|v| if v.is_nan() { 0.0 } else { *v })
            .collect::<Vec<_>>(),
        s,
    );
    let a1 = ema_of(&apc, r);
    let a2 = ema_of(
        &a1.iter()
            .map(|v| if v.is_nan() { 0.0 } else { *v })
            .collect::<Vec<_>>(),
        s,
    );
    for i in 0..n {
        if !e2[i].is_nan() && a2[i].abs() > 1e-12 {
            tsi_line[i] = 100.0 * e2[i] / a2[i];
        }
    }
    let sig_src: Vec<f64> = tsi_line
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    let sig_all = ema_of(&sig_src, 7);
    for i in 0..n {
        if !tsi_line[i].is_nan() && i >= 7 && !sig_all[i].is_nan() {
            signal[i] = sig_all[i];
        }
    }
    (tsi_line, signal)
}

/// Coppock Curve: WMA10(ROC14) + WMA11(ROC11).
pub fn coppock(series: &OhlcvSeries) -> Vec<f64> {
    use crate::roc;
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if n <= 30 {
        return out;
    }
    let r14 = roc(series, 14);
    let r11 = roc(series, 11);
    let w10 = wma_of(&r14, 10);
    let w11 = wma_of(&r11, 11);
    for i in 0..n {
        if !w10[i].is_nan() && !w11[i].is_nan() {
            out[i] = w10[i] + w11[i];
        }
    }
    out
}

fn wma_of(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n < period {
        return out;
    }
    let denom = (period * (period + 1) / 2) as f64;
    for i in (period - 1)..n {
        let w = &values[i + 1 - period..=i];
        if w.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mut acc = 0.0;
        for (k, v) in w.iter().enumerate() {
            acc += v * (k + 1) as f64;
        }
        out[i] = acc / denom;
    }
    out
}

/// Weighted Moving Average of closes.
pub fn wma(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    wma_of(&closes(series), period)
}

/// Hull Moving Average.
pub fn hull_ma(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || n < period {
        return out;
    }
    let c = closes(series);
    let half = (period / 2).max(1);
    let sq = (period as f64).sqrt() as usize;
    let w_half = wma_of(&c, half);
    let w_full = wma_of(&c, period);
    let mut diff = vec![f64::NAN; n];
    for i in 0..n {
        if !w_half[i].is_nan() && !w_full[i].is_nan() {
            diff[i] = 2.0 * w_half[i] - w_full[i];
        }
    }
    let raw = wma_of(
        &diff
            .iter()
            .map(|v| if v.is_nan() { 0.0 } else { *v })
            .collect::<Vec<_>>(),
        sq.max(1),
    );
    for i in 0..n {
        if !diff[i].is_nan() && !raw[i].is_nan() {
            // raw was computed over zero-filled prefix; only trust it once the
            // window is fully defined.
            if i + 1 >= period + sq {
                out[i] = raw[i];
            }
        }
    }
    out
}

/// Arnaud Legoux Moving Average.
pub fn alma(series: &OhlcvSeries, period: usize, offset: f64, sigma: f64) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n < period || sigma <= 0.0 {
        return out;
    }
    let c = closes(series);
    let m = offset * (period as f64 - 1.0);
    let s = sigma;
    let mut weights = vec![0.0; period];
    let mut wsum = 0.0;
    for i in 0..period {
        let w = (-(i as f64 - m).powi(2) / (2.0 * s * s)).exp();
        weights[i] = w;
        wsum += w;
    }
    for w in weights.iter_mut() {
        *w /= wsum;
    }
    for i in (period - 1)..n {
        out[i] = weights
            .iter()
            .zip(&c[i + 1 - period..=i])
            .map(|(w, v)| w * v)
            .sum();
    }
    out
}

/// Kaufman Adaptive Moving Average.
pub fn kama(series: &OhlcvSeries, period: usize, fast: usize, slow: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    let c = closes(series);
    let fast_sc = 2.0 / (fast.max(1) as f64 + 1.0);
    let slow_sc = 2.0 / (slow.max(1) as f64 + 1.0);
    let mut prev = c[period - 1];
    out[period - 1] = prev;
    for i in period..n {
        let direction = (c[i] - c[i - period]).abs();
        let volatility: f64 = (1..=period).map(|k| (c[i + 1 - k] - c[i - k]).abs()).sum();
        let er = if volatility > 1e-12 {
            direction / volatility
        } else {
            0.0
        };
        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);
        prev += sc * (c[i] - prev);
        out[i] = prev;
    }
    out
}

/// Multiple SMAs: (20, 50, 200).
pub fn multi_sma(series: &OhlcvSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let c = closes(series);
    (sma_of(&c, 20), sma_of(&c, 50), sma_of(&c, 200))
}

/// Multiple EMAs: (12, 26, 50).
pub fn multi_ema(series: &OhlcvSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let c = closes(series);
    (ema_of(&c, 12), ema_of(&c, 26), ema_of(&c, 50))
}

/// Ultimate Oscillator (0-100).
pub fn ultimate_osc(series: &OhlcvSeries, p1: usize, p2: usize, p3: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if p1 == 0 || p2 == 0 || p3 == 0 || n <= 2 * p3 {
        return out;
    }
    let mut bp = vec![0.0; n];
    let mut tr = vec![0.0; n];
    for i in 1..n {
        let c = &series.candles[i];
        let p = &series.candles[i - 1];
        bp[i] = c.close - c.low.min(p.close);
        tr[i] = (c.high - c.low)
            .max((c.high - p.close).abs())
            .max((c.low - p.close).abs());
    }
    let avg = |period: usize| -> Vec<f64> {
        let mut v = vec![f64::NAN; n];
        let mut sbp = 0.0;
        let mut str_ = 0.0;
        for i in 0..n {
            sbp += bp[i];
            str_ += tr[i];
            if i >= period {
                sbp -= bp[i - period];
                str_ -= tr[i - period];
            }
            if i + 1 >= period {
                v[i] = if str_ > 1e-12 { sbp / str_ } else { 0.5 };
            }
        }
        v
    };
    let a1 = avg(p1);
    let a2 = avg(p2);
    let a3 = avg(p3);
    for i in 0..n {
        if !a1[i].is_nan() && !a2[i].is_nan() && !a3[i].is_nan() {
            out[i] = 100.0 * (4.0 * a1[i] + 2.0 * a2[i] + a3[i]) / 7.0;
        }
    }
    out
}

/// Detrended Price Oscillator: close minus SMA shifted back period/2 + 1.
pub fn dpo(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    let c = closes(series);
    let sma_vals = sma_of(&c, period);
    let shift = period / 2 + 1;
    for i in 0..n {
        if i + shift < n && !sma_vals[i + shift].is_nan() {
            out[i] = c[i] - sma_vals[i + shift];
        }
    }
    out
}

/// Natural-log returns; first bar is NaN.
pub fn log_returns(series: &OhlcvSeries) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    for i in 1..n {
        let prev = series.candles[i - 1].close;
        if prev > 0.0 && series.candles[i].close > 0.0 {
            out[i] = (series.candles[i].close / prev).ln();
        }
    }
    out
}

/// Know Sure Thing with its signal line.
#[allow(clippy::too_many_arguments)]
pub fn kst(
    series: &OhlcvSeries,
    r1: usize,
    r2: usize,
    r3: usize,
    r4: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    sig: usize,
) -> (Vec<f64>, Vec<f64>) {
    use crate::roc;
    let n = series.candles.len();
    let mut kst_line = vec![f64::NAN; n];
    let mut signal = vec![f64::NAN; n];
    if r1 == 0 || r2 == 0 || r3 == 0 || r4 == 0 || n <= r4 + a4 + sig {
        return (kst_line, signal);
    }
    let decent = |v: &[f64]| {
        v.iter()
            .map(|x| if x.is_nan() { 0.0 } else { *x })
            .collect::<Vec<_>>()
    };
    let rc1 = decent(&roc(series, r1));
    let rc2 = decent(&roc(series, r2));
    let rc3 = decent(&roc(series, r3));
    let rc4 = decent(&roc(series, r4));
    let s1 = sma_of(&rc1, a1);
    let s2 = sma_of(&rc2, a2);
    let s3 = sma_of(&rc3, a3);
    let s4 = sma_of(&rc4, a4);
    for i in 0..n {
        if !s1[i].is_nan() && !s2[i].is_nan() && !s3[i].is_nan() && !s4[i].is_nan() {
            kst_line[i] = s1[i] + 2.0 * s2[i] + 3.0 * s3[i] + 4.0 * s4[i];
        }
    }
    let sig_all = sma_of(
        &kst_line
            .iter()
            .map(|v| if v.is_nan() { 0.0 } else { *v })
            .collect::<Vec<_>>(),
        sig,
    );
    for i in 0..n {
        if !kst_line[i].is_nan() && !sig_all[i].is_nan() {
            signal[i] = sig_all[i];
        }
    }
    (kst_line, signal)
}

/// Elder Ray: (bull power, bear power) vs EMA-13.
pub fn elder_ray(series: &OhlcvSeries, period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = series.candles.len();
    let mut bull = vec![f64::NAN; n];
    let mut bear = vec![f64::NAN; n];
    if period == 0 || n < period {
        return (bull, bear);
    }
    let e = ema_of(&closes(series), period);
    for i in 0..n {
        if !e[i].is_nan() {
            bull[i] = series.candles[i].high - e[i];
            bear[i] = series.candles[i].low - e[i];
        }
    }
    (bull, bear)
}

/// Vortex Indicator: (VI+, VI-).
pub fn vortex(series: &OhlcvSeries, period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = series.candles.len();
    let mut plus = vec![f64::NAN; n];
    let mut minus = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return (plus, minus);
    }
    let mut pvm = vec![0.0; n];
    let mut mvm = vec![0.0; n];
    let mut tr = vec![0.0; n];
    for i in 1..n {
        let c = &series.candles[i];
        let p = &series.candles[i - 1];
        pvm[i] = (c.high - p.low).abs();
        mvm[i] = (c.low - p.high).abs();
        tr[i] = (c.high - c.low)
            .max((c.high - p.close).abs())
            .max((c.low - p.close).abs());
    }
    for i in period..n {
        let sp: f64 = pvm[i + 1 - period..=i].iter().sum();
        let sm: f64 = mvm[i + 1 - period..=i].iter().sum();
        let st: f64 = tr[i + 1 - period..=i].iter().sum();
        if st > 1e-12 {
            plus[i] = sp / st;
            minus[i] = sm / st;
        }
    }
    (plus, minus)
}

/// Aroon Up / Down (0-100).
pub fn aroon(series: &OhlcvSeries, period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = series.candles.len();
    let mut up = vec![f64::NAN; n];
    let mut down = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return (up, down);
    }
    for i in period..n {
        let w = &series.candles[i + 1 - period..=i];
        let mut hi_idx = 0;
        let mut lo_idx = 0;
        for (k, c) in w.iter().enumerate() {
            if c.high >= w[hi_idx].high {
                hi_idx = k;
            }
            if c.low <= w[lo_idx].low {
                lo_idx = k;
            }
        }
        up[i] = 100.0 * hi_idx as f64 / period as f64;
        down[i] = 100.0 * lo_idx as f64 / period as f64;
    }
    (up, down)
}

/// Aroon Oscillator: Aroon Up minus Aroon Down (-100..100).
pub fn aroon_osc(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let (up, down) = aroon(series, period);
    up.iter()
        .zip(down.iter())
        .map(|(u, d)| {
            if u.is_nan() || d.is_nan() {
                f64::NAN
            } else {
                u - d
            }
        })
        .collect()
}

/// Money Flow Index (0-100).
pub fn mfi(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    let mut pmf = vec![0.0; n];
    let mut nmf = vec![0.0; n];
    for i in 1..n {
        let c = &series.candles[i];
        let p = &series.candles[i - 1];
        let tp = (c.high + c.low + c.close) / 3.0;
        let ptp = (p.high + p.low + p.close) / 3.0;
        let mf = tp * c.volume.max(0.0);
        if tp > ptp {
            pmf[i] = mf;
        } else if tp < ptp {
            nmf[i] = mf;
        }
    }
    for i in period..n {
        let ps: f64 = pmf[i + 1 - period..=i].iter().sum();
        let ns: f64 = nmf[i + 1 - period..=i].iter().sum();
        out[i] = if ns <= 1e-12 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + ps / ns)
        };
    }
    out
}

/// Cumulative Money Flow Volume: typical price × volume, summed.
pub fn mfv(series: &OhlcvSeries) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    let mut acc = 0.0;
    for (i, c) in series.candles.iter().enumerate() {
        acc += (c.high + c.low + c.close) / 3.0 * c.volume.max(0.0);
        out[i] = acc;
    }
    out
}

/// Price Volume Trend (cumulative).
pub fn pvt(series: &OhlcvSeries) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let mut acc = 0.0;
    out[0] = 0.0;
    for i in 1..n {
        let prev = series.candles[i - 1].close;
        if prev.abs() > 1e-12 {
            acc += (series.candles[i].close - prev) / prev * series.candles[i].volume.max(0.0);
        }
        out[i] = acc;
    }
    out
}

/// Accumulation/Distribution line (cumulative CLV × volume).
pub fn ad_line(series: &OhlcvSeries) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    let mut acc = 0.0;
    for (i, c) in series.candles.iter().enumerate() {
        let denom = c.high - c.low;
        let clv = if denom.abs() > 1e-12 {
            ((c.close - c.low) - (c.high - c.close)) / denom
        } else {
            0.0
        };
        acc += clv * c.volume.max(0.0);
        out[i] = acc;
    }
    out
}

/// Ease of Movement (SMA-smoothed, raw scale, no magic multiplier).
pub fn eom(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    let mut raw = vec![0.0; n];
    for i in 1..n {
        let c = &series.candles[i];
        let p = &series.candles[i - 1];
        let dm = (c.high + c.low) / 2.0 - (p.high + p.low) / 2.0;
        let box_ratio = if (c.high - c.low).abs() > 1e-12 {
            c.volume.max(0.0) / (c.high - c.low)
        } else {
            0.0
        };
        raw[i] = if box_ratio.abs() > 1e-12 {
            dm / box_ratio
        } else {
            0.0
        };
    }
    let sm = sma_of(&raw, period);
    for i in 0..n {
        if i >= period && !sm[i].is_nan() {
            out[i] = sm[i];
        }
    }
    out
}

/// Force Index: EMA of (close change × volume).
pub fn force_index(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return out;
    }
    let mut raw = vec![0.0; n];
    for i in 1..n {
        raw[i] = (series.candles[i].close - series.candles[i - 1].close)
            * series.candles[i].volume.max(0.0);
    }
    let sm = ema_of(&raw, period);
    for i in 0..n {
        if i >= period && !sm[i].is_nan() {
            out[i] = sm[i];
        }
    }
    out
}

/// Mass Index: 25-bar sum of single/double EMA9 range ratios.
pub fn mass_index(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n <= 2 * 9 + period {
        return out;
    }
    let range: Vec<f64> = series
        .candles
        .iter()
        .map(|c| (c.high - c.low).max(0.0))
        .collect();
    let single = ema_of(&range, 9);
    let dbl = ema_of(
        &single
            .iter()
            .map(|v| if v.is_nan() { 0.0 } else { *v })
            .collect::<Vec<_>>(),
        9,
    );
    for i in 0..n {
        if i + 1 < period || single[i].is_nan() || dbl[i].abs() < 1e-12 {
            continue;
        }
        let mut acc = 0.0;
        let mut ok = true;
        for k in (i + 1 - period)..=i {
            if single[k].is_nan() || dbl[k].abs() < 1e-12 {
                ok = false;
                break;
            }
            acc += single[k] / dbl[k];
        }
        if ok {
            out[i] = acc;
        }
    }
    out
}

/// Realized volatility: stdev of log returns × sqrt(252) × 100 (annualized %).
pub fn realized_vol(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || n <= period {
        return out;
    }
    let lr = log_returns(series);
    for i in period..n {
        let w: Vec<f64> = lr[i + 1 - period..=i].to_vec();
        if w.iter().any(|v| v.is_nan()) {
            continue;
        }
        let mean = w.iter().sum::<f64>() / period as f64;
        let var = w.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / period as f64;
        out[i] = var.sqrt() * (252.0_f64).sqrt() * 100.0;
    }
    out
}

/// Rolling z-score of close vs SMA-20 in units of rolling stdev.
pub fn zscore(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period < 2 || n < period {
        return out;
    }
    let c = closes(series);
    let sma_vals = sma_of(&c, period);
    let sd_vals = stdev_of(&c, period);
    for i in 0..n {
        if !sma_vals[i].is_nan() && sd_vals[i] > 1e-12 {
            out[i] = (c[i] - sma_vals[i]) / sd_vals[i];
        } else if !sma_vals[i].is_nan() {
            out[i] = 0.0;
        }
    }
    out
}

/// VWAP bands: VWAP ± k × rolling-20 stdev of typical price.
pub fn vwap_bands(series: &OhlcvSeries, k: f64) -> (Vec<f64>, Vec<f64>) {
    use crate::vwap;
    let n = series.candles.len();
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    if n < 20 {
        return (upper, lower);
    }
    let v = vwap(series);
    let tp: Vec<f64> = series
        .candles
        .iter()
        .map(|c| (c.high + c.low + c.close) / 3.0)
        .collect();
    let sd = stdev_of(&tp, 20);
    for i in 0..n {
        if !v[i].is_nan() && !sd[i].is_nan() {
            upper[i] = v[i] + k * sd[i];
            lower[i] = v[i] - k * sd[i];
        }
    }
    (upper, lower)
}

/// Keltner Channel width: (upper − lower) / middle.
pub fn keltner_width(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    use crate::keltner;
    let (middle, upper, lower) = keltner(series, period, 2.0);
    middle
        .iter()
        .zip(upper.iter().zip(lower.iter()))
        .map(|(m, (u, l))| {
            if m.is_nan() || u.is_nan() || l.is_nan() || m.abs() < 1e-12 {
                f64::NAN
            } else {
                (u - l) / m
            }
        })
        .collect()
}

/// Highest-high / lowest-low channel over `period`.
pub fn high_low_band(series: &OhlcvSeries, period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = series.candles.len();
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    if period == 0 || n < period {
        return (upper, lower);
    }
    for i in (period - 1)..n {
        let w = &series.candles[i + 1 - period..=i];
        upper[i] = w.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
        lower[i] = w.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    }
    (upper, lower)
}

/// Ulcer Index (%): sqrt of mean squared percent drawdown over `period`.
pub fn ulcer(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n < period {
        return out;
    }
    let c = closes(series);
    for i in (period - 1)..n {
        let w = &c[i + 1 - period..=i];
        let peak = w.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        if peak <= 0.0 {
            continue;
        }
        let mean_sq = w
            .iter()
            .map(|v| (100.0 * (v - peak) / peak).powi(2))
            .sum::<f64>()
            / period as f64;
        out[i] = mean_sq.sqrt();
    }
    out
}

/// Trend Intensity (0-100): share of closes above SMA(`period`) in the window.
/// Documented definition: a breadth-of-trend gauge, not anyone's trademark.
pub fn trend_intensity(series: &OhlcvSeries, period: usize) -> Vec<f64> {
    let n = series.candles.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 || n < 2 * period {
        return out;
    }
    let c = closes(series);
    let sma_vals = sma_of(&c, period);
    for i in (2 * period - 1)..n {
        // Only defined when the whole window has SMA values.
        if ((i + 1 - period)..=i).all(|k| !sma_vals[k].is_nan()) {
            let above = ((i + 1 - period)..=i)
                .filter(|&k| c[k] > sma_vals[k])
                .count();
            out[i] = 100.0 * above as f64 / period as f64;
        }
    }
    out
}

/// Supertrend: (trailing line, direction +1/-1).
pub fn supertrend(series: &OhlcvSeries, period: usize, mult: f64) -> (Vec<f64>, Vec<f64>) {
    use crate::atr;
    let n = series.candles.len();
    let mut line = vec![f64::NAN; n];
    let mut dir = vec![f64::NAN; n];
    if period == 0 || n <= period {
        return (line, dir);
    }
    let atr_vals = atr(series, period);
    let c: Vec<f64> = closes(series);
    let median: Vec<f64> = series
        .candles
        .iter()
        .map(|cd| (cd.high + cd.low) / 2.0)
        .collect();
    // First defined index.
    let start = match atr_vals.iter().position(|v| !v.is_nan()) {
        Some(i) => i,
        None => return (line, dir),
    };
    let mut fub = median[start] + mult * atr_vals[start];
    let mut flb = median[start] - mult * atr_vals[start];
    let mut trend = 1.0;
    line[start] = flb;
    dir[start] = trend;
    for i in (start + 1)..n {
        if atr_vals[i].is_nan() {
            line[i] = line[i - 1];
            dir[i] = trend;
            continue;
        }
        let basic_ub = median[i] + mult * atr_vals[i];
        let basic_lb = median[i] - mult * atr_vals[i];
        fub = if basic_ub < fub || c[i - 1] > fub {
            basic_ub.min(fub)
        } else {
            fub
        };
        flb = if basic_lb > flb || c[i - 1] < flb {
            basic_lb.max(flb)
        } else {
            flb
        };
        if trend == 1.0 && c[i] < flb {
            trend = -1.0;
        } else if trend == -1.0 && c[i] > fub {
            trend = 1.0;
        }
        dir[i] = trend;
        line[i] = if trend == 1.0 { flb } else { fub };
    }
    (line, dir)
}

/// Classic floor pivot levels from a prior high/low/close.
#[derive(Debug, Clone, Copy)]
pub struct PivotLevels {
    pub pp: f64,
    pub r1: f64,
    pub r2: f64,
    pub r3: f64,
    pub s1: f64,
    pub s2: f64,
    pub s3: f64,
}

pub fn pivot_levels(high: f64, low: f64, close: f64) -> PivotLevels {
    let pp = (high + low + close) / 3.0;
    PivotLevels {
        pp,
        r1: 2.0 * pp - low,
        s1: 2.0 * pp - high,
        r2: pp + (high - low),
        s2: pp - (high - low),
        r3: high + 2.0 * (pp - low),
        s3: low - 2.0 * (high - pp),
    }
}

/// Fibonacci retracement levels between a swing low and high.
pub fn fib_levels(low: f64, high: f64) -> Vec<(f64, f64)> {
    let range = high - low;
    [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        .iter()
        .map(|r| (*r, low + range * r))
        .collect()
}

// === EXTENDED PART 2 TESTS APPENDED BELOW ===

#[cfg(test)]
mod tests_part2 {
    use super::*;
    use bt_core::{Candle, OhlcvSeries};

    fn series(n: usize) -> OhlcvSeries {
        let candles = (0..n)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.4 + 2.0 * ((i as f64 * 0.6).sin());
                Candle::new(
                    i as f64 * 86_400.0,
                    base - 1.0,
                    base + 1.5,
                    base - 1.5,
                    base,
                    1_000_000.0 + i as f64 * 5_000.0,
                )
            })
            .collect();
        OhlcvSeries::new("T", candles)
    }

    fn finite_after_warmup(v: &[f64], warmup: usize) {
        assert_eq!(v.len(), 150);
        assert!(v[..warmup].iter().all(|x| x.is_nan() || x.is_finite()));
        assert!(v[warmup..].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_volume_family_shapes() {
        let s = series(150);
        finite_after_warmup(&mfi(&s, 14), 14);
        finite_after_warmup(&mfv(&s), 0);
        finite_after_warmup(&pvt(&s), 1);
        finite_after_warmup(&ad_line(&s), 0);
        finite_after_warmup(&eom(&s, 14), 15);
        finite_after_warmup(&force_index(&s, 13), 14);
        finite_after_warmup(&mass_index(&s, 25), 45);
        assert!(mfi(&s, 14)[20..].iter().all(|x| (0.0..=100.0).contains(x)));
        // Rising prices with rising volume: PVT climbs.
        assert!(pvt(&s).last().copied().unwrap() > 0.0);
    }

    #[test]
    fn test_volatility_shapes_and_bounds() {
        let s = series(150);
        finite_after_warmup(&realized_vol(&s, 20), 21);
        finite_after_warmup(&zscore(&s, 20), 19);
        finite_after_warmup(&keltner_width(&s, 20), 20);
        finite_after_warmup(&ulcer(&s, 14), 13);
        let (up, lo) = high_low_band(&s, 20);
        assert_eq!((up.len(), lo.len()), (150, 150));
        for (i, c) in s.candles.iter().enumerate().skip(19) {
            assert!(up[i] >= c.high - 1e-9 && lo[i] <= c.low + 1e-9);
        }
        let (vu, vl) = vwap_bands(&s, 1.0);
        assert!(vu[30] >= vl[30]);
        // Flat series: realized vol and ulcer vanish, z-score is zero.
        let flat = OhlcvSeries::new(
            "F",
            (0..60)
                .map(|i| Candle::new(i as f64, 50.0, 50.0, 50.0, 50.0, 1000.0))
                .collect(),
        );
        assert!((realized_vol(&flat, 20)[59]).abs() < 1e-9);
        assert!((ulcer(&flat, 14)[59]).abs() < 1e-9);
    }

    #[test]
    fn test_trend_overlays_follow_price() {
        let s = series(260);
        let (line, dir) = supertrend(&s, 10, 3.0);
        assert_eq!((line.len(), dir.len()), (260, 260));
        assert!(dir[259] == 1.0 || dir[259] == -1.0);
        // Strong ramp ends up-trending.
        let ramp = OhlcvSeries::new(
            "R",
            (0..120)
                .map(|i| {
                    let b = 10.0 + i as f64;
                    Candle::new(i as f64, b - 0.2, b + 0.2, b - 0.2, b, 1000.0)
                })
                .collect(),
        );
        let (_, d) = supertrend(&ramp, 10, 3.0);
        assert_eq!(d[119], 1.0);
        for f in [
            kama(&s, 10, 2, 30),
            alma(&s, 9, 0.85, 6.0),
            hull_ma(&s, 9),
            wma(&s, 14),
        ] {
            assert_eq!(f.len(), 260);
            assert!(f[259].is_finite());
        }
        let (a, b, c) = multi_sma(&s);
        assert_eq!((a.len(), b.len(), c.len()), (260, 260, 260));
        assert!(a[259].is_finite() && c[259].is_finite());
        // Perfect ramp: everything reads full trend intensity.
        assert!((trend_intensity(&ramp, 30)[119] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_scalar_levels_are_exact() {
        let p = pivot_levels(110.0, 90.0, 100.0);
        assert!((p.pp - 100.0).abs() < 1e-9);
        assert!((p.r1 - 110.0).abs() < 1e-9);
        assert!((p.s1 - 90.0).abs() < 1e-9);
        assert!((p.r2 - 120.0).abs() < 1e-9);
        assert!((p.s2 - 80.0).abs() < 1e-9);
        let fibs = fib_levels(90.0, 110.0);
        assert_eq!(fibs.len(), 7);
        assert!((fibs[0].1 - 90.0).abs() < 1e-9);
        assert!((fibs[6].1 - 110.0).abs() < 1e-9);
        assert!((fibs[3].1 - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_oscillator_bounds() {
        let s = series(150);
        assert!(stoch_rsi(&s, 14, 14)[30..]
            .iter()
            .all(|x| (0.0..=100.0).contains(x)));
        let (t, sig) = tsi(&s, 25, 13);
        assert!(t[60..].iter().all(|x| x.is_finite() && x.abs() <= 100.0));
        assert!(sig[80..].iter().all(|x| x.is_finite()));
        assert!(kama(&s, 10, 2, 30)[100].is_finite());
        let (up, down) = aroon(&s, 14);
        assert!(up[30..].iter().all(|x| (0.0..=100.0).contains(x)));
        assert!(down[30..].iter().all(|x| (0.0..=100.0).contains(x)));
        assert!(aroon_osc(&s, 14)[30..]
            .iter()
            .all(|x| (-100.0..=100.0).contains(x)));
        let (k, ks) = kst(&s, 10, 15, 20, 30, 10, 10, 10, 15, 9);
        assert!(k[120..].iter().all(|x| x.is_finite()));
        assert!(ks[130..].iter().all(|x| x.is_finite()));
        let u = ultimate_osc(&s, 7, 14, 28);
        assert!(u[70..].iter().all(|x| (0.0..=100.0).contains(x)));
        let d = dpo(&s, 20);
        // DPO is centered: the trailing shift bars are undefined by design.
        assert!(d[100..139].iter().all(|x| x.is_finite()));
        assert!(d[139..].iter().all(|x| x.is_nan()));
        let lr = log_returns(&s);
        assert!(lr[0].is_nan() && lr[1..].iter().all(|x| x.is_finite()));
        let (eb, er) = elder_ray(&s, 13);
        assert!(eb[100] > 0.0, "ramp bull power must be positive");
        let _ = er;
        let (vp, vm) = vortex(&s, 14);
        assert!(vp[100] > vm[100], "ramp favors +VI");
        let cp = coppock(&s);
        assert!(cp[100..].iter().all(|x| x.is_finite()));
    }
}

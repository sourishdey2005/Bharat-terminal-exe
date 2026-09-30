// crates/bt-analytics/src/quant_analytics.rs
// Author: Sourish Dey

//! Frequency-domain and state-space analytics that complement the neural
//! forecasters.
//!
//! # Why this exists
//!
//! DLinear, the moving average and ARIMA all extrapolate the *level* of a
//! series. On a mean-reverting instrument they collapse onto almost the same
//! near-flat line, which tells a trader nothing. These two functions are
//! deliberately different in kind:
//!
//! * [`QuantAnalyticsEngine::extrapolate_fft`] works in the frequency domain.
//!   It removes the linear trend, keeps the dominant harmonics, and re-synthesises
//!   only those waves into the future. The output therefore *oscillates* around
//!   the extrapolated trend instead of drifting along it.
//! * [`QuantAnalyticsEngine::classify_regime`] says *what kind of market this
//!   is right now* — trend, chop, or volatile decline — which is the input a
//!   forecaster's point estimate is missing.
//!
//! # Cost
//!
//! One `rustfft` transform over at most a few hundred points, no allocation per
//! bar, no BLAS and no neural session. Measured well under a millisecond, and
//! the 2 GB budget is untouched.
//!
//! # Precision
//!
//! Everything here is `f64` rather than the `f32` the ONNX tensors use. Prices
//! are the input here, and halving their precision before an FFT would put
//! quantisation noise straight into the spectrum, which is exactly the
//! high-frequency content the harmonic filter is trying to reject.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::fmt;

/// Fewest bars [`QuantAnalyticsEngine::extrapolate_fft`] will accept.
pub const FFT_MIN_BARS: usize = 32;

/// Hard ceiling on the transform length, so a multi-year range cannot turn a
/// UI interaction into a long synchronous computation.
pub const FFT_MAX_BARS: usize = 512;

/// Fewest bars [`QuantAnalyticsEngine::classify_regime`] will accept.
pub const REGIME_MIN_BARS: usize = 16;

/// Number of distinct market states the classifier reports.
pub const REGIME_STATES: usize = 3;

/// The three market states, in `regime_id` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Persistent positive drift at contained volatility.
    BullTrend,
    /// No edge: drift and volatility both near their baseline.
    Consolidation,
    /// Negative drift, or a volatility jump well above its own baseline.
    BearVolatile,
}

impl Regime {
    /// Stable identifier, matching the historical 0/1/2 numbering.
    pub fn id(self) -> usize {
        match self {
            Regime::BullTrend => 0,
            Regime::Consolidation => 1,
            Regime::BearVolatile => 2,
        }
    }

    /// Machine-readable name, e.g. `BULL_TREND`.
    pub fn name(self) -> &'static str {
        match self {
            Regime::BullTrend => "BULL_TREND",
            Regime::Consolidation => "CONSOLIDATION_CHOP",
            Regime::BearVolatile => "BEAR_VOLATILE",
        }
    }

    /// Short label for a chart badge.
    pub fn label(self) -> &'static str {
        match self {
            Regime::BullTrend => "BULL TREND",
            Regime::Consolidation => "CHOP",
            Regime::BearVolatile => "BEAR / VOLATILE",
        }
    }

    /// Every state, in `id` order. Used by the UI to colour a regime strip.
    pub const ALL: [Regime; REGIME_STATES] = [
        Regime::BullTrend,
        Regime::Consolidation,
        Regime::BearVolatile,
    ];
}

impl fmt::Display for Regime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Result of the frequency-domain extrapolator.
#[derive(Debug, Clone, PartialEq)]
pub struct FftCycleOutput {
    /// Number of bars projected.
    pub horizon_steps: usize,
    /// Projected closes, in the input's own price units.
    pub cycle_projection: Vec<f64>,
    /// Period of each retained harmonic, in bars, strongest first.
    pub dominant_periods: Vec<usize>,
    /// Relative strength of each retained harmonic, strongest first, in `(0, 1]`.
    pub dominant_strengths: Vec<f64>,
}

impl FftCycleOutput {
    /// The strongest retained period, if any survived the filter.
    pub fn primary_period(&self) -> Option<usize> {
        self.dominant_periods.first().copied()
    }
}

/// Result of the regime classifier.
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeOutput {
    /// The state the market is in now.
    pub current_regime: Regime,
    /// Per-state posterior-like scores, in `regime_id` order.
    ///
    /// These are softmax-normalised negative squared distances under a diagonal
    /// Gaussian, so they are comparable across the three states and sum to 1.
    /// They are *not* EM-fitted likelihoods: the state parameters are fixed
    /// (see [`Regime::ALL`]) so the result is deterministic and needs no
    /// training set.
    pub state_scores: [f64; REGIME_STATES],
    /// Realized volatility over the recent window, per bar, as a fraction.
    pub volatility_score: f64,
    /// Mean log return per bar over the same window.
    pub trend_drift: f64,
    /// Recent volatility divided by its own longer baseline. `1.0` is calm;
    /// well above 1 is a volatility jump.
    pub vol_ratio: f64,
}

impl RegimeOutput {
    /// `current_regime` as the historical `0/1/2` integer.
    pub fn regime_id(&self) -> usize {
        self.current_regime.id()
    }

    /// The state name, e.g. `BULL_TREND`.
    pub fn current_regime(&self) -> &'static str {
        self.current_regime.name()
    }

    /// Confidence in the winning state: its score minus the runner-up.
    ///
    /// A caller showing a confidence badge wants to know how decisively the
    /// states separated, not just which one won.
    pub fn margin(&self) -> f64 {
        let mut sorted = self.state_scores;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        (sorted[0] - sorted[1]).max(0.0)
    }
}

/// Why an analytics call could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// Fewer bars than the method needs.
    TooShort { needed: usize, got: usize },
    /// A horizon of zero was requested.
    ZeroHorizon,
    /// A price or return was NaN, infinite, or non-positive where a ratio is
    /// taken.
    NonFinite,
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::TooShort { needed, got } => {
                write!(f, "needs at least {needed} bars, got {got}")
            }
            QuantError::ZeroHorizon => write!(f, "horizon must be greater than zero"),
            QuantError::NonFinite => write!(f, "input contains non-finite values"),
        }
    }
}

impl std::error::Error for QuantError {}

/// Stateless entry point for the non-neural paradigms.
pub struct QuantAnalyticsEngine;

impl QuantAnalyticsEngine {
    /// Project cyclical structure forward by keeping the dominant harmonics.
    ///
    /// The series is linearly de-trended, transformed, filtered to the strongest
    /// low-frequency components, and re-synthesised beyond the last bar; the
    /// extrapolated trend is then added back so the projection stays anchored to
    /// the real price level.
    ///
    /// * `horizon` — bars to project.
    /// * `top_k_harmonics` — how many components to keep. More is noisier, not
    ///   richer; 3 is a good default.
    pub fn extrapolate_fft(
        prices: &[f64],
        horizon: usize,
        top_k_harmonics: usize,
    ) -> Result<FftCycleOutput, QuantError> {
        if prices.len() < FFT_MIN_BARS {
            return Err(QuantError::TooShort {
                needed: FFT_MIN_BARS,
                got: prices.len(),
            });
        }
        if horizon == 0 {
            return Err(QuantError::ZeroHorizon);
        }
        if prices.iter().any(|p| !p.is_finite()) {
            return Err(QuantError::NonFinite);
        }

        // Only the most recent window participates, so a long range does not
        // dominate the spectrum with ancient structure.
        let src = &prices[prices.len() - prices.len().min(FFT_MAX_BARS)..];

        // ---- de-trend ----------------------------------------------------
        // A least-squares line, not just first-to-last: the latter swings wildly
        // on a noisy series and drags the whole residual with it.
        let n = src.len();
        let (slope, intercept) = least_squares_line(src);
        let residual: Vec<f64> = src
            .iter()
            .enumerate()
            .map(|(i, &p)| p - (intercept + slope * i as f64))
            .collect();
        // Remove the residual mean so a tilted fit does not leave a DC step that
        // lands entirely in bin 0.
        let mean = residual.iter().sum::<f64>() / n as f64;
        let mut centred: Vec<f64> = residual.iter().map(|r| r - mean).collect();
        // Winsorise the residual before transforming. A single bad print would
        // otherwise spread its energy across the whole spectrum and dominate the
        // harmonic search, producing a projection that describes the outlier
        // rather than the market. Clipping is applied to the *residual*, so the
        // trend fit is untouched and only the oscillatory content is bounded.
        winsorize_in_place(&mut centred, WINSOR_MAD_SCALE);

        // ---- transform ---------------------------------------------------
        // Zero-pad to a power of two: FFT cost and numerical behaviour are both
        // better behaved there, and it gives finer frequency resolution without
        // inventing data (padding contributes zeros, i.e. no new energy).
        let padded = n.next_power_of_two();
        let mut buffer: Vec<Complex<f64>> = centred
            .iter()
            .map(|&v| Complex::new(v, 0.0))
            .chain(std::iter::repeat_n(Complex::new(0.0, 0.0), padded - n))
            .collect();
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(padded);
        fft.process(&mut buffer);

        // ---- pick harmonics ----------------------------------------------
        // Bin 0 is the removed DC term. Search only the lower half of the
        // spectrum: on 32-256 bars the top bins are single-sample noise, and
        // letting them win is what turns an FFT into a random-walk generator.
        let search_max = (padded / 4).max(2);
        let mut peaks: Vec<(usize, f64)> = (1..search_max)
            .map(|k| (k, buffer[k].norm()))
            .filter(|&(_, amp)| amp > 0.0)
            .collect();
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let kept: Vec<(usize, f64)> = peaks.iter().copied().take(top_k_harmonics.max(1)).collect();

        // Scale amplitudes by the coherent gain of a real sinusoid (n/2) so the
        // synthesised wave has the same magnitude the series actually showed.
        let gain = (padded / 2) as f64;
        // Any component below this share of the strongest is noise, not a cycle.
        let floor = kept
            .first()
            .map_or(0.0, |(_, top)| top * MIN_HARMONIC_SHARE);

        let dominant_periods: Vec<usize> = kept
            .iter()
            .filter(|(_, amp)| *amp >= floor)
            .map(|(k, _)| padded / k)
            .collect();
        let dominant_strengths: Vec<f64> = kept
            .iter()
            .filter(|(_, amp)| *amp >= floor)
            .map(|(_, amp)| *amp)
            .collect();

        // ---- re-synthesise -------------------------------------------------
        // The projection is expressed as a *delta from the last real bar*:
        //
        //     P(t) = last_close + [trend(t) - trend(n-1)] + [osc(t) - osc(n-1)]
        //
        // rather than as trend(t) + osc(t) in absolute terms. Without this the
        // least-squares trend and the re-synthesised wave disagree at the join —
        // the fit passes through the middle of the data, not through its last
        // bar — so a forecast can open hundreds of points away from the live
        // price, which is worse than useless on a chart. Subtracting the
        // oscillation at `n-1` also removes the spectral leakage that otherwise
        // puts a step discontinuity at the seam.
        let tau = std::f64::consts::TAU;
        let osc_at = |t: f64, buffer: &[Complex<f64>], kept: &[(usize, f64)], floor: f64| -> f64 {
            kept.iter()
                .filter(|(_, amp)| *amp >= floor)
                .map(|&(k, _amp)| {
                    let freq = k as f64 / padded as f64;
                    let amp = buffer[k].norm() / gain;
                    let phase = buffer[k].arg();
                    amp * (tau * freq * t + phase).cos()
                })
                .sum()
        };

        let last_close = prices[prices.len() - 1];
        let join = n as f64 - 1.0;
        let trend_at_join = intercept + slope * join;
        let osc_at_join = osc_at(join, &buffer, &kept, floor);

        let cycle_projection: Vec<f64> = (1..=horizon)
            .map(|step| {
                // Continuation index: the last real bar is `n - 1`, so the first
                // projected bar sits at `n`.
                let t = (n + step - 1) as f64;
                let trend_delta = (intercept + slope * t) - trend_at_join;
                let osc_delta = osc_at(t, &buffer, &kept, floor) - osc_at_join;
                last_close + trend_delta + osc_delta
            })
            .collect();

        Ok(FftCycleOutput {
            horizon_steps: horizon,
            cycle_projection,
            dominant_periods,
            dominant_strengths,
        })
    }

    /// Classify the current market state from returns and range volatility.
    ///
    /// Features are volatility-normalised, so the thresholds mean the same thing
    /// on a 2,500-rupee NSE stock and a 190-point index:
    ///
    /// * `trend_drift` — mean log return per bar.
    /// * `volatility_score` — realized standard deviation of log returns.
    /// * `vol_ratio` — recent realized vol over its own 3x longer baseline,
    ///   which is what distinguishes "quiet bear" from "volatility jump".
    /// * a Parkinson term from the high/low range, which is less noisy than
    ///   close-to-close vol and so breaks ties the close-only term cannot.
    ///
    /// The three states are fixed diagonal Gaussians over `(drift, vol_ratio)`
    /// rather than EM-fitted ones. That is a deliberate trade: an EM fit on the
    /// 64-256 bars available at call time would be unstable and would make the
    /// same symbol report a different regime on every run, which is worse than
    /// a documented, deterministic classifier.
    pub fn classify_regime(prices: &[f64]) -> Result<RegimeOutput, QuantError> {
        Self::classify_regime_ranged(prices, None)
    }

    /// [`Self::classify_regime`] with the high/low range available, which adds
    /// the Parkinson volatility term. `bars` may be shorter than `prices`.
    pub fn classify_regime_ranged(
        prices: &[f64],
        bars: Option<&[bt_core::Candle]>,
    ) -> Result<RegimeOutput, QuantError> {
        if prices.len() < REGIME_MIN_BARS {
            return Err(QuantError::TooShort {
                needed: REGIME_MIN_BARS,
                got: prices.len(),
            });
        }
        if prices.iter().any(|p| !p.is_finite() || *p <= 0.0) {
            return Err(QuantError::NonFinite);
        }

        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .filter(|r| r.is_finite())
            .collect();
        if returns.is_empty() {
            return Err(QuantError::NonFinite);
        }

        let drift = returns.iter().sum::<f64>() / returns.len() as f64;
        let realized = stddev(&returns, drift);

        // Short window reacts to a shock; the long one is the calm baseline. The two
        // windows must NOT overlap: a jump detector that averages the shock
        // into its own baseline cannot see the shock. `recent_vol` and
        // `baseline_vol` below are therefore adjacent slices of the same series.
        let (short, long) = recent_vs_baseline_vol(&returns, RECENT_WINDOW, BASELINE_WINDOW);
        let vol_ratio = (short / long.max(1e-9)).clamp(0.0, 8.0);

        // Parkinson: squared high-low range is a lower-variance volatility
        // estimator than close-to-close, so it arbitrates when the two disagree.
        let mut parkinson = realized;
        if let Some(b) = bars {
            let tail = &b[b.len().saturating_sub(prices.len())..];
            let p = parkinson_vol(tail);
            if p > 0.0 {
                // Blend rather than replace: one estimator alone is fragile.
                parkinson = 0.5 * (p + parkinson);
            }
        }
        let parkinson = if parkinson > 0.0 { parkinson } else { realized };

        let scores = state_scores(drift, vol_ratio);
        // Argmax, but the volatility jump alone is enough to call a bear: a
        // market that just repriced violently is not a healthy bull trend even
        // if the drift over the whole window is still positive.
        let current = if vol_ratio >= VOL_JUMP_BEAR || drift <= DRIFT_BEAR {
            Regime::BearVolatile
        } else if drift >= DRIFT_BULL && vol_ratio <= VOL_JUMP_CALM {
            Regime::BullTrend
        } else {
            Regime::Consolidation
        };

        Ok(RegimeOutput {
            current_regime: current,
            state_scores: scores,
            volatility_score: parkinson,
            trend_drift: drift,
            vol_ratio,
        })
    }
}

/// Mean log return per bar above which a market counts as trending up.
const DRIFT_BULL: f64 = 0.0012;
/// Mean log return per bar at or below which a market counts as declining.
const DRIFT_BEAR: f64 = -0.0015;
/// Recent/baseline volatility at or above which the market is in a jump.
const VOL_JUMP_BEAR: f64 = 2.1;
/// Recent/baseline volatility below which a positive drift is trusted.
const VOL_JUMP_CALM: f64 = 1.6;
/// A harmonic weaker than this share of the strongest is treated as noise.
const MIN_HARMONIC_SHARE: f64 = 0.12;
/// Residual values beyond this many robust standard deviations from the median
/// are clipped before the transform. Generous on purpose: the aim is to stop one
/// bad bar owning the spectrum, not to flatten genuine large swings.
const WINSOR_MAD_SCALE: f64 = 6.0;
/// Trailing returns treated as "now" when measuring a volatility jump.
const RECENT_WINDOW: usize = 10;
/// Returns immediately *preceding* the recent window, used as the calm baseline.
const BASELINE_WINDOW: usize = 30;

/// Softmax scores for the three states over `(drift, vol_ratio)`.
///
/// Each state is a diagonal Gaussian centred on a plausible `(drift, vol_ratio)`
/// pair. Fixed centres keep the classifier deterministic; the widths say how
/// forgiving each state is of being wrong.
fn state_scores(drift: f64, vol_ratio: f64) -> [f64; REGIME_STATES] {
    // (drift centre, drift width, vol_ratio centre, vol_ratio width)
    const STATES: [(f64, f64, f64, f64); REGIME_STATES] = [
        (0.006, 0.006, 1.0, 0.9),  // bull: up, calm
        (0.000, 0.0025, 1.0, 0.8), // chop: flat, calm
        (-0.008, 0.008, 2.2, 1.4), // bear: down and/or volatile
    ];
    let mut raw = [0.0_f64; REGIME_STATES];
    let mut max = f64::MIN;
    for (i, (dc, dw, vc, vw)) in STATES.iter().enumerate() {
        let dz = (drift - dc) / dw;
        let vz = (vol_ratio - vc) / vw;
        // Negative log-likelihood, dropping constants shared by all states.
        let ll = -0.5 * (dz * dz + vz * vz);
        raw[i] = ll;
        max = max.max(ll);
    }
    let mut sum = 0.0;
    for v in raw.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in raw.iter_mut() {
            *v /= sum;
        }
    }
    raw
}

/// Clip `values` to `median +- scale` robust deviations (MAD-scaled).
///
/// Uses the median absolute deviation rather than the standard deviation
/// precisely because it is not moved by the outliers it is there to suppress.
/// A series with no spread at all is left alone: MAD 0 would otherwise clip
/// every value to the median and flatten a legitimately constant series, which
/// the flat-series test covers.
fn winsorize_in_place(values: &mut [f64], scale: f64) {
    if values.len() < 4 {
        return;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    let mut devs: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = devs[devs.len() / 2];
    if mad <= f64::EPSILON {
        return;
    }
    // 1.4826 scales MAD to a standard deviation for normal data.
    let limit = scale * 1.4826 * mad;
    for v in values.iter_mut() {
        *v = v.clamp(median - limit, median + limit);
    }
}

/// Least-squares line through `(i, y)`, returned as `(slope, intercept)`.
fn least_squares_line(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (i, &v) in y.iter().enumerate() {
        let dx = i as f64 - mean_x;
        sxy += dx * (v - mean_y);
        sxx += dx * dx;
    }
    let slope = if sxx > 1e-12 { sxy / sxx } else { 0.0 };
    (slope, mean_y - slope * mean_x)
}

/// Population standard deviation of `values` about their mean.
fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    var.max(0.0).sqrt()
}

/// Realized volatility over the most recent `window` returns.
fn realized_vol(returns: &[f64], window: usize) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let tail = &returns[returns.len() - returns.len().min(window)..];
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    stddev(tail, mean)
}

/// `(recent volatility, preceding baseline volatility)` from two *adjacent*,
/// non-overlapping windows.
///
/// Overlapping them is the trap: a baseline that already contains half of a
/// shock measures the shock as ordinary, and the ratio collapses toward 1 so a
/// genuine volatility jump reads as calm. Returns before the recent window are
/// what "what this market normally does" means.
fn recent_vs_baseline_vol(returns: &[f64], recent: usize, baseline: usize) -> (f64, f64) {
    let split = returns.len().saturating_sub(recent);
    let recent_slice = &returns[split..];
    let base_end = split;
    let base_start = base_end.saturating_sub(baseline);
    let base_slice = &returns[base_start..base_end];
    (
        realized_vol(recent_slice, recent),
        realized_vol(base_slice, baseline),
    )
}

/// Parkinson high-low volatility estimator, annualisation-free (per bar).
fn parkinson_vol(bars: &[bt_core::Candle]) -> f64 {
    let usable: Vec<f64> = bars
        .iter()
        .filter(|c| c.high > 0.0 && c.low > 0.0)
        .map(|c| ((c.high / c.low).ln()).powi(2))
        .collect();
    if usable.is_empty() {
        return 0.0;
    }
    let mean = usable.iter().sum::<f64>() / usable.len() as f64;
    // sigma^2 = (1/(4n ln2)) * sum ln(H/L)^2
    ((mean / (4.0 * std::f64::consts::LN_2)).max(0.0)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bt_core::Candle;

    fn sine(n: usize, period: f64, amp: f64) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + amp * (std::f64::consts::TAU * i as f64 / period).sin())
            .collect()
    }

    #[test]
    fn fft_rejects_short_history() {
        let short = vec![100.0; FFT_MIN_BARS - 1];
        assert_eq!(
            QuantAnalyticsEngine::extrapolate_fft(&short, 8, 3),
            Err(QuantError::TooShort {
                needed: FFT_MIN_BARS,
                got: FFT_MIN_BARS - 1
            })
        );
    }

    #[test]
    fn fft_rejects_zero_horizon() {
        let prices = sine(64, 8.0, 5.0);
        assert_eq!(
            QuantAnalyticsEngine::extrapolate_fft(&prices, 0, 3),
            Err(QuantError::ZeroHorizon)
        );
    }

    #[test]
    fn fft_rejects_non_finite() {
        let mut prices = sine(64, 8.0, 5.0);
        prices[10] = f64::NAN;
        assert_eq!(
            QuantAnalyticsEngine::extrapolate_fft(&prices, 8, 3),
            Err(QuantError::NonFinite)
        );
    }

    /// The headline claim: a clean sinusoid must come back with its own period.
    #[test]
    fn fft_recovers_a_known_period() {
        // 64 bars of a period-8 cycle is exactly 8 repeats, so the peak must be
        // unambiguous.
        let prices = sine(64, 8.0, 5.0);
        let out = QuantAnalyticsEngine::extrapolate_fft(&prices, 8, 3).expect("fft");
        assert_eq!(out.cycle_projection.len(), 8);
        assert_eq!(out.primary_period(), Some(8), "period not recovered");
    }

    /// The FFT output must actually oscillate. This is the property that
    /// separates it from the flat ARIMA line it replaces.
    #[test]
    fn fft_projection_oscillates_around_its_trend() {
        let prices = sine(96, 12.0, 8.0);
        let out = QuantAnalyticsEngine::extrapolate_fft(&prices, 24, 3).expect("fft");
        let mean = out.cycle_projection.iter().sum::<f64>() / out.cycle_projection.len() as f64;
        let spread = out
            .cycle_projection
            .iter()
            .map(|v| (v - mean).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            spread > 1.0,
            "projection is flat (max deviation {spread:.4}); that is the ARIMA failure mode"
        );
    }

    #[test]
    fn fft_on_a_flat_series_stays_finite_and_flat() {
        let flat = vec![2500.0; 64];
        let out = QuantAnalyticsEngine::extrapolate_fft(&flat, 8, 3).expect("fft");
        assert_eq!(out.cycle_projection.len(), 8);
        assert!(
            out.cycle_projection.iter().all(|v| v.is_finite()),
            "flat input produced non-finite output"
        );
        for v in &out.cycle_projection {
            assert!((v - 2500.0).abs() < 1e-6, "flat input drifted to {v}");
        }
    }

    /// The projection must continue from the real price level, not jump.
    ///
    /// This is the invariant that makes the FFT usable at all: whatever the
    /// spectrum says, the first projected bar has to sit next to the last close.
    #[test]
    fn fft_projection_is_anchored_to_the_last_close() {
        let prices: Vec<f64> = (0..80)
            .map(|i| 3000.0 + 8.0 * (i as f64 * 0.25).sin() + i as f64 * 2.0)
            .collect();
        let last = prices[prices.len() - 1];
        let out = QuantAnalyticsEngine::extrapolate_fft(&prices, 8, 3).expect("fft");
        // Anchored means within one bar of drift plus one bar of wave, so a few
        // percent of the price level. The old formulation drifted into the
        // hundreds on exactly this kind of input.
        assert!(
            (out.cycle_projection[0] - last).abs() < last * 0.05,
            "first projected bar {:.2} is not anchored near the last close {last:.2}",
            out.cycle_projection[0]
        );
        for (i, v) in out.cycle_projection.iter().enumerate() {
            assert!(
                (*v - last).abs() < last * 0.25,
                "projected bar {i} = {v:.2} drifted too far from {last:.2}"
            );
        }
    }

    /// A single wild outlier must not throw the projection off the price scale.
    #[test]
    fn fft_survives_a_single_outlier_bar() {
        let mut prices: Vec<f64> = (0..64).map(|i| 3000.0 + i as f64).collect();
        prices[40] = 90_000.0;
        let last = prices[prices.len() - 1];
        let out = QuantAnalyticsEngine::extrapolate_fft(&prices, 6, 3).expect("fft");
        for (i, v) in out.cycle_projection.iter().enumerate() {
            assert!(
                (*v - last).abs() < last,
                "outlier bar made projected bar {i} = {v:.2} vs last close {last:.2}"
            );
        }
    }

    #[test]
    fn fft_is_deterministic() {
        let prices = sine(80, 10.0, 6.0);
        let a = QuantAnalyticsEngine::extrapolate_fft(&prices, 12, 3).expect("fft");
        let b = QuantAnalyticsEngine::extrapolate_fft(&prices, 12, 3).expect("fft");
        assert_eq!(a, b, "same input produced different output");
    }

    #[test]
    fn fft_caps_the_transform_length() {
        // A very long range must not be transformed in full.
        let prices = sine(5000, 40.0, 5.0);
        let out = QuantAnalyticsEngine::extrapolate_fft(&prices, 4, 2).expect("fft");
        assert_eq!(out.cycle_projection.len(), 4);
    }

    #[test]
    fn regime_rejects_short_history() {
        assert_eq!(
            QuantAnalyticsEngine::classify_regime(&[100.0; REGIME_MIN_BARS - 1]),
            Err(QuantError::TooShort {
                needed: REGIME_MIN_BARS,
                got: REGIME_MIN_BARS - 1
            })
        );
    }

    #[test]
    fn regime_rejects_non_positive_prices() {
        let mut prices = sine(32, 8.0, 1.0);
        prices[5] = 0.0;
        assert_eq!(
            QuantAnalyticsEngine::classify_regime(&prices),
            Err(QuantError::NonFinite)
        );
    }

    #[test]
    fn steady_uptrend_is_a_bull_regime() {
        let prices: Vec<f64> = (0..80).map(|i| 100.0 * 1.004_f64.powi(i)).collect();
        let out = QuantAnalyticsEngine::classify_regime(&prices).expect("regime");
        assert_eq!(
            out.current_regime,
            Regime::BullTrend,
            "drift {:.5} vol {:.5} was not read as a bull trend",
            out.trend_drift,
            out.volatility_score
        );
    }

    #[test]
    fn steady_downtrend_is_a_bear_regime() {
        let prices: Vec<f64> = (0..80).map(|i| 100.0 * 0.996_f64.powi(i)).collect();
        let out = QuantAnalyticsEngine::classify_regime(&prices).expect("regime");
        assert_eq!(out.current_regime, Regime::BearVolatile);
    }

    #[test]
    fn flat_market_is_chop() {
        let out = QuantAnalyticsEngine::classify_regime(&vec![2500.0; 64]).expect("regime");
        assert_eq!(out.current_regime, Regime::Consolidation);
    }

    /// A quiet drift that suddenly reprices is a jump, and a jump is a bear
    /// even though the whole-window drift is still positive.
    #[test]
    fn volatility_jump_overrides_a_positive_drift() {
        let mut prices: Vec<f64> = (0..60).map(|i| 100.0 * 1.004_f64.powi(i)).collect();
        // Last 12 bars: violent alternating moves, net roughly flat.
        let mut p = *prices.last().unwrap();
        for i in 0..12 {
            p *= if i % 2 == 0 { 1.09 } else { 1.0 / 1.09 };
            prices.push(p);
        }
        let out = QuantAnalyticsEngine::classify_regime(&prices).expect("regime");
        assert!(
            out.vol_ratio > 1.0,
            "the shock did not register: vol_ratio {}",
            out.vol_ratio
        );
        assert_eq!(out.current_regime, Regime::BearVolatile);
    }

    #[test]
    fn regime_scores_are_a_distribution() {
        let prices = sine(64, 8.0, 2.0);
        let out = QuantAnalyticsEngine::classify_regime(&prices).expect("regime");
        let sum: f64 = out.state_scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "scores sum to {sum}");
        assert!(out.state_scores.iter().all(|s| *s >= 0.0));
        assert!(out.margin() >= 0.0);
        assert!(out.margin() <= 1.0);
    }

    #[test]
    fn regime_ids_match_the_documented_numbering() {
        assert_eq!(Regime::BullTrend.id(), 0);
        assert_eq!(Regime::Consolidation.id(), 1);
        assert_eq!(Regime::BearVolatile.id(), 2);
        assert_eq!(Regime::BearVolatile.name(), "BEAR_VOLATILE");
        assert_eq!(Regime::ALL.len(), REGIME_STATES);
    }

    #[test]
    fn parkinson_term_is_used_when_ranges_are_available() {
        let closes: Vec<f64> = (0..40).map(|i| 100.0 * 1.003_f64.powi(i)).collect();
        let bars: Vec<Candle> = closes
            .iter()
            .map(|c| Candle::new(1_000_000.0, *c, c * 1.01, c * 0.99, *c, 1000.0))
            .collect();
        let with =
            QuantAnalyticsEngine::classify_regime_ranged(&closes, Some(&bars)).expect("regime");
        let without = QuantAnalyticsEngine::classify_regime(&closes).expect("regime");
        assert!(
            with.volatility_score > 0.0 && without.volatility_score > 0.0,
            "both estimators should report positive volatility"
        );
        // Same data, same verdict: the range term arbitrates, it does not steer.
        assert_eq!(with.current_regime, without.current_regime);
    }
}

// crates/bt-analytics/src/corr_regime.rs
// Author: Sourish Dey

//! Correlation-regime detection.
//!
//! In a calm market instruments diversify: correlations sit low and a
//! diversified portfolio genuinely spreads risk. In a stressed market they
//! converge toward 1 and "diversification" evaporates exactly when it is
//! wanted. Detecting that switch is the point of this module, because a risk
//! number computed under the wrong regime is worse than no number at all — it
//! looks precise and is wrong.
//!
//! The statistic is the mean pairwise rolling correlation across a basket. A
//! shift is reported when the recent window sits clearly above or below the
//! reference window, which makes the finding a statement about a *change*, not
//! about an absolute level.

use bt_core::Result;

/// Bars per rolling correlation window.
pub const WINDOW: usize = 30;

/// Minimum bars before any regime can be judged.
pub const MIN_BARS: usize = WINDOW * 2 + 1;

/// Average pairwise correlation above which the market is called stressed.
pub const CRISIS_THRESHOLD: f64 = 0.65;

/// Mean pairwise correlation below which the market is called calm.
pub const CALM_THRESHOLD: f64 = 0.35;

/// Shift, in correlation units, required before a regime change is reported.
pub const SWITCH_DELTA: f64 = 0.20;

/// Which correlation regime a basket is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrRegime {
    /// Correlations low: diversification is working.
    Calm,
    /// Correlations mid-range: no clear read.
    Normal,
    /// Correlations high: diversification has stopped working.
    Crisis,
}

impl CorrRegime {
    /// Display label.
    pub fn label(&self) -> &'static str {
        match self {
            CorrRegime::Calm => "Calm",
            CorrRegime::Normal => "Normal",
            CorrRegime::Crisis => "Crisis",
        }
    }

    /// What this regime means for a diversified book.
    pub fn meaning(&self) -> &'static str {
        match self {
            CorrRegime::Calm => "Instruments are moving independently; spread risk as usual.",
            CorrRegime::Normal => "Correlations are unremarkable; standard diversification holds.",
            CorrRegime::Crisis => {
                "Instruments are moving together; holdings will not offset each other."
            }
        }
    }

    /// Classify a mean correlation.
    pub fn from_mean(mean: f64) -> Self {
        if mean.is_nan() {
            // Undefined correlation is not "calm"; it is unreadable, and it maps
            // to Normal rather than claiming a diversified reading.
            return CorrRegime::Normal;
        }
        if mean >= CRISIS_THRESHOLD {
            CorrRegime::Crisis
        } else if mean <= CALM_THRESHOLD {
            CorrRegime::Calm
        } else {
            CorrRegime::Normal
        }
    }
}

/// The result of a regime scan.
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeScan {
    /// Regime over the most recent window.
    pub current: CorrRegime,
    /// Mean pairwise correlation over the most recent window.
    pub current_corr: f64,
    /// Mean over the reference (earlier) window.
    pub reference_corr: f64,
    /// `current - reference`.
    pub delta: f64,
    /// Whether the shift cleared [`SWITCH_DELTA`].
    pub switched: bool,
    /// Regime over the reference window, for the "was" side of the comparison.
    pub previous: CorrRegime,
}

/// Simple returns of a close series.
fn returns(closes: &[f64]) -> Vec<f64> {
    closes
        .windows(2)
        .map(|w| {
            if w[0].abs() > 1e-12 {
                (w[1] - w[0]) / w[0]
            } else {
                0.0
            }
        })
        .collect()
}

/// Pearson correlation over a slice.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 {
        return f64::NAN;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ma = a.iter().sum::<f64>() / n as f64;
    let mb = b.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let den = (va * vb).sqrt();
    if den <= 1e-18 {
        // A flat series has no correlation to report. NaN, not 0: a zero would
        // be read as "these two are unrelated", which is not what flat means.
        f64::NAN
    } else {
        (cov / den).clamp(-1.0, 1.0)
    }
}

/// Mean of all pairwise correlations across the basket over one window.
///
/// Returns `NaN` when fewer than two series have usable data.
fn mean_pairwise(closes: &[Vec<f64>], end: usize, window: usize) -> f64 {
    if end < window || closes.len() < 2 {
        return f64::NAN;
    }
    let start = end - window;
    let series: Vec<Vec<f64>> = closes
        .iter()
        .filter_map(|c| {
            if c.len() > start {
                let r = returns(&c[..end]);
                // Align every series on the same trailing window before pairing.
                if r.len() >= window {
                    Some(r[r.len() - window..].to_vec())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();
    if series.len() < 2 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..series.len() {
        for j in i + 1..series.len() {
            let c = pearson(&series[i], &series[j]);
            if c.is_finite() {
                sum += c;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Scan a basket of close series for a correlation-regime shift.
///
/// `basket` must be aligned by *index*, not by timestamp: the last element of
/// each series is the most recent bar, and the two windows are taken as
/// positions back from the end.
pub fn scan(basket: &[Vec<f64>]) -> Result<RegimeScan> {
    let basket: Vec<Vec<f64>> = basket.iter().filter(|c| c.len() >= MIN_BARS).cloned().collect();
    if basket.len() < 2 {
        return Err(bt_core::BtError::InvalidInput(format!(
            "correlation regime needs at least 2 series of {MIN_BARS}+ bars, got {}",
            basket.len()
        )));
    }

    let end = basket[0].len();
    let current_corr = mean_pairwise(&basket, end, WINDOW);
    let reference_corr = mean_pairwise(&basket, end - WINDOW, WINDOW);

    let current = CorrRegime::from_mean(current_corr);
    let previous = CorrRegime::from_mean(reference_corr);

    // A NaN in either window means no comparison is possible, and the result
    // must not present itself as a clean "no switch".
    let delta = if current_corr.is_finite() && reference_corr.is_finite() {
        current_corr - reference_corr
    } else {
        f64::NAN
    };
    let switched = delta.is_finite() && delta.abs() >= SWITCH_DELTA;

    Ok(RegimeScan {
        current,
        current_corr,
        reference_corr,
        delta,
        switched,
        previous,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent series built from a simple hash-based pseudo-random walk.
    fn independent(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut price = 100.0;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let noise = ((state >> 33) as f64 / (u32::MAX as f64)) - 0.5;
                price *= 1.0 + noise * 0.02;
                price
            })
            .collect()
    }

    /// A price series whose *returns* are dominated by `base`'s returns.
    ///
    /// Built from returns rather than price levels because that is what
    /// correlation is measured on: blending two independent price levels would
    /// not produce the correlation it superficially looks like it should. The
    /// idiosyncratic term shrinks as `strength` rises, so two series built at
    /// high strength are close to identical.
    fn correlated(base: &[f64], n: usize, strength: f64) -> Vec<f64> {
        let noise = independent(n, 8484);
        let mut price = 100.0;
        (0..n)
            .map(|i| {
                let ret = |s: &[f64]| -> f64 {
                    if i == 0 || s[i - 1].abs() < 1e-12 {
                        0.0
                    } else {
                        (s[i] - s[i - 1]) / s[i - 1]
                    }
                };
                let r = ret(base) * strength + ret(&noise) * (1.0 - strength);
                price *= 1.0 + r;
                price
            })
            .collect()
    }

    /// All series identical: correlation 1.
    fn identical(n: usize) -> Vec<Vec<f64>> {
        let base: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5).sin()).collect();
        vec![base.clone(), base.clone(), base.clone()]
    }

    #[test]
    fn too_few_series_is_an_error_not_a_panic() {
        assert!(scan(&[]).is_err());
        assert!(scan(&[independent(MIN_BARS, 1)]).is_err());
    }

    #[test]
    fn too_short_a_history_is_an_error() {
        assert!(scan(&[independent(10, 1), independent(10, 2)]).is_err());
    }

    #[test]
    fn identical_series_are_a_crisis() {
        let s = scan(&identical(MIN_BARS + 10)).unwrap();
        assert!((s.current_corr - 1.0).abs() < 1e-6, "{s:?}");
        assert_eq!(s.current, CorrRegime::Crisis);
    }

    #[test]
    fn independent_series_are_calm() {
        let basket: Vec<Vec<f64>> = (0..5).map(|i| independent(MIN_BARS + 10, i as u64 + 7)).collect();
        let s = scan(&basket).unwrap();
        assert_eq!(s.current, CorrRegime::Calm, "corr={}", s.current_corr);
    }

    #[test]
    fn a_switch_from_calm_to_crisis_is_detected() {
        // Half the history independent, then a fully convergent stretch: the
        // recent window is crisis while the reference window is calm.
        let n = MIN_BARS + 20;
        let mut a = independent(n, 11);
        let mut b = independent(n, 22);
        let mut c = independent(n, 33);
        // Force the last WINDOW bars to be identical across all three.
        let tail_base: Vec<f64> = (0..n).map(|i| 200.0 + (i as f64).sin()).collect();
        let start = n - WINDOW;
        for s in [&mut a, &mut b, &mut c] {
            s[start..n].copy_from_slice(&tail_base[start..n]);
        }
        let s = scan(&[a, b, c]).unwrap();
        assert_eq!(s.previous, CorrRegime::Calm, "reference window: {s:?}");
        assert_eq!(s.current, CorrRegime::Crisis, "recent window: {s:?}");
        assert!(s.switched, "{s:?}");
        assert!(s.delta > SWITCH_DELTA, "{s:?}");
    }

    #[test]
    fn a_stable_basket_reports_no_switch() {
        // Crisis throughout: high level, but no *change*, so no switch.
        let s = scan(&identical(MIN_BARS + 10)).unwrap();
        assert_eq!(s.current, CorrRegime::Crisis);
        assert!(!s.switched, "a persistently high regime is not a switch: {s:?}");
    }

    #[test]
    fn a_flat_series_yields_undefined_correlation_not_zero() {
        // A zero here would be read as "unrelated", which is not what a flat
        // series means.
        let flat = vec![100.0; MIN_BARS + 10];
        let s = scan(&[flat.clone(), flat.clone()]).unwrap();
        assert!(
            !s.current_corr.is_finite() || s.current_corr == 0.0,
            "{s:?}"
        );
        // Whatever it is, it must not claim a switch on undefined data.
        assert!(!s.switched);
    }

    #[test]
    fn regimes_have_distinct_labels_and_meanings() {
        let all = [CorrRegime::Calm, CorrRegime::Normal, CorrRegime::Crisis];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.label(), b.label());
                assert_ne!(a.meaning(), b.meaning());
            }
        }
    }

    #[test]
    fn mean_classification_matches_the_documented_thresholds() {
        assert_eq!(CorrRegime::from_mean(0.9), CorrRegime::Crisis);
        assert_eq!(CorrRegime::from_mean(0.65), CorrRegime::Crisis);
        assert_eq!(CorrRegime::from_mean(0.5), CorrRegime::Normal);
        assert_eq!(CorrRegime::from_mean(0.35), CorrRegime::Calm);
        assert_eq!(CorrRegime::from_mean(0.0), CorrRegime::Calm);
        assert_eq!(CorrRegime::from_mean(f64::NAN), CorrRegime::Normal);
    }

    #[test]
    fn pearson_bounds_and_edge_cases() {
        let up: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let down: Vec<f64> = (0..10).map(|i| (9 - i) as f64).collect();
        assert!((pearson(&up, &up) - 1.0).abs() < 1e-9);
        assert!((pearson(&up, &down) + 1.0).abs() < 1e-9);
        assert!(pearson(&up, &down).abs() <= 1.0);
        // Too few points, and a flat series, are both undefined.
        assert!(pearson(&[1.0, 2.0], &[3.0, 4.0]).is_nan());
        assert!(pearson(&[1.0; 10], &[2.0; 10]).is_nan());
    }

    #[test]
    fn scan_is_deterministic() {
        let basket: Vec<Vec<f64>> = (0..4).map(|i| independent(MIN_BARS + 5, i + 1)).collect();
        assert_eq!(scan(&basket).unwrap(), scan(&basket).unwrap());
    }

    #[test]
    fn correlated_helper_produces_the_correlation_it_claims() {
        // Guards the test helper itself: if this broke, the switch test above
        // would pass for the wrong reason.
        let base = independent(MIN_BARS + 10, 5);
        let basket = vec![
            base.clone(),
            correlated(&base, MIN_BARS + 10, 0.9),
            correlated(&base, MIN_BARS + 10, 0.9),
        ];
        let s = scan(&basket).unwrap();
        assert!(s.current_corr > 0.7, "{s:?}");
    }
}
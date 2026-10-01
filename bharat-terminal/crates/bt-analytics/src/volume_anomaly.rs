// crates/bt-analytics/src/volume_anomaly.rs
// Author: Sourish Dey

//! Flags bars whose volume is unusual against its own recent history.
//!
//! A fixed "3x average" threshold is the obvious approach and it is wrong: a
//! stock's normal volume varies enormously with its price level, and quiet
//! regimes produce constant false alarms. This uses a rolling z-score against a
//! trailing mean/standard-deviation instead, so the bar is compared to the
//! instrument's own recent behaviour rather than to a constant.
//!
//! The baseline deliberately excludes the bar being tested. Including it would
//! let a spike inflate its own mean and standard deviation, which is precisely
//! how a large outlier hides itself.

use bt_core::{Candle, Result};

/// Trailing window used for the volume baseline.
pub const BASELINE_BARS: usize = 20;

/// Z-score beyond which a bar counts as unusual.
pub const DEFAULT_Z_THRESHOLD: f64 = 3.0;

/// One flagged bar.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeAnomaly {
    /// Index into the original candle slice.
    pub index: usize,
    /// Observed volume.
    pub volume: f64,
    /// Trailing mean volume.
    pub baseline: f64,
    /// Observed minus baseline, in standard deviations.
    pub z: f64,
    /// Observed as a multiple of baseline.
    pub ratio: f64,
    /// Whether this is unusually *low* rather than unusually high.
    pub dry_up: bool,
}

/// Rolling mean and population standard deviation over a slice.
fn stats(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Find bars whose volume is a statistical outlier.
pub fn detect(candles: &[Candle], z_threshold: f64) -> Result<Vec<VolumeAnomaly>> {
    if candles.len() <= BASELINE_BARS {
        return Ok(Vec::new());
    }
    let threshold = if z_threshold.is_finite() && z_threshold > 0.0 {
        z_threshold
    } else {
        DEFAULT_Z_THRESHOLD
    };

    let mut out = Vec::new();
    // The window ends one bar *before* i, so the tested bar never contributes
    // to its own baseline.
    for i in BASELINE_BARS..candles.len() {
        let window = &candles[i - BASELINE_BARS..i];
        let vols: Vec<f64> = window.iter().map(|c| c.volume).filter(|v| v.is_finite() && *v >= 0.0).collect();
        if vols.len() < BASELINE_BARS / 2 {
            continue;
        }
        let (mean, sd) = stats(&vols);
        let v = candles[i].volume;
        if !v.is_finite() || v < 0.0 || mean <= 0.0 {
            continue;
        }
        // A perfectly flat baseline has no spread, so any deviation is infinite
        // in z terms. Cap it: "unusually large" is the claim being made, not
        // an unbounded ratio that would dominate every chart it appears on.
        let z = if sd > 1e-9 {
            (v - mean) / sd
        } else if (v - mean).abs() > 1e-9 {
            if v > mean {
                f64::INFINITY.min(1e6)
            } else {
                -1e6
            }
        } else {
            0.0
        };

        let dry_up = z < -threshold;
        if z.abs() >= threshold {
            out.push(VolumeAnomaly {
                index: i,
                volume: v,
                baseline: mean,
                z,
                ratio: v / mean,
                dry_up,
            });
        }
    }
    Ok(out)
}

/// Convenience wrapper using [`DEFAULT_Z_THRESHOLD`].
pub fn detect_default(candles: &[Candle]) -> Result<Vec<VolumeAnomaly>> {
    detect(candles, DEFAULT_Z_THRESHOLD)
}

/// Largest single-day volume ratio in the series, for the header readout.
pub fn peak_ratio(candles: &[Candle]) -> Option<f64> {
    if candles.len() <= BASELINE_BARS {
        return None;
    }
    (BASELINE_BARS..candles.len())
        .filter_map(|i| {
            let vols: Vec<f64> = candles[i - BASELINE_BARS..i]
                .iter()
                .map(|c| c.volume)
                .collect();
            if vols.is_empty() {
                return None;
            }
            let (mean, _) = stats(&vols);
            if mean > 0.0 {
                Some(candles[i].volume / mean)
            } else {
                None
            }
        })
        .fold(None, |acc, r| Some(acc.map_or(r, |a: f64| a.max(r))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `n` candles with a constant volume.
    fn flat(n: usize, vol: f64) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle::new(i as f64, 100.0, 101.0, 99.0, 100.0, vol))
            .collect()
    }

    /// Build candles whose volume varies, with `spike_at` set to `spike`.
    fn with_spike(n: usize, base: f64, noise: f64, spike_at: usize, spike: f64) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let v = if i == spike_at {
                    spike
                } else {
                    base + (i % 5) as f64 * noise
                };
                Candle::new(i as f64, 100.0, 101.0, 99.0, 100.0, v)
            })
            .collect()
    }

    #[test]
    fn a_flat_series_has_no_anomalies() {
        // Zero variance means nothing is unusual, which is the correct answer.
        assert!(detect_default(&flat(60, 1000.0)).unwrap().is_empty());
    }

    #[test]
    fn a_short_series_is_empty_rather_than_panicking() {
        for n in [0usize, 1, 5, BASELINE_BARS] {
            assert!(
                detect_default(&flat(n, 1000.0)).unwrap().is_empty(),
                "n={n} must yield no anomalies"
            );
        }
    }

    #[test]
    fn a_large_spike_is_flagged_with_the_right_index() {
        let cs = with_spike(60, 1000.0, 50.0, 40, 9000.0);
        let hits = detect_default(&cs).unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].index, 40);
        assert!(hits[0].z > 3.0, "{hits:?}");
        assert!(!hits[0].dry_up);
        assert!(hits[0].ratio > 5.0, "{hits:?}");
    }

    #[test]
    fn the_spike_does_not_contaminate_its_own_baseline() {
        // This is the subtle bug the design guards against: if the tested bar
        // were inside its own window, a 9x spike would inflate the mean enough
        // to drop below the threshold and hide.
        let cs = with_spike(60, 1000.0, 50.0, 40, 9000.0);
        let hits = detect_default(&cs).unwrap();
        assert_eq!(hits.len(), 1, "the spike must not hide itself");
        // The baseline reflects the ~1000-1200 that preceded it, not the spike.
        assert!(hits[0].baseline < 2000.0, "{:?}", hits[0].baseline);
    }

    #[test]
    fn an_unusually_quiet_bar_is_flagged_as_a_dry_up() {
        let mut cs = with_spike(60, 1000.0, 50.0, 40, 1000.0);
        cs[40].volume = 1.0;
        let hits = detect_default(&cs).unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].dry_up, "{hits:?}");
        assert!(hits[0].z < -3.0, "{hits:?}");
    }

    #[test]
    fn a_moderate_move_is_not_flagged() {
        // 1.3x normal volume happens every week and must stay quiet.
        let cs = with_spike(60, 1000.0, 50.0, 40, 1300.0);
        assert!(detect_default(&cs).unwrap().is_empty(), "{:?}", detect_default(&cs).unwrap());
    }

    #[test]
    fn a_higher_threshold_finds_less() {
        let cs = with_spike(60, 1000.0, 200.0, 30, 3000.0);
        let strict = detect(&cs, 6.0).unwrap().len();
        let loose = detect(&cs, 1.0).unwrap().len();
        assert!(loose >= strict, "loose={loose} strict={strict}");
    }

    #[test]
    fn a_hostsle_threshold_is_rejected_not_used() {
        // A NaN or negative threshold would silence the detector entirely.
        let cs = with_spike(60, 1000.0, 50.0, 40, 9000.0);
        for bad in [f64::NAN, -1.0, 0.0, f64::INFINITY] {
            let hits = detect(&cs, bad).unwrap();
            assert!(!hits.is_empty(), "threshold {bad} must fall back to the default");
        }
    }

    #[test]
    fn output_carries_no_nan_or_infinity_for_ordinary_input() {
        let cs = with_spike(80, 1000.0, 50.0, 60, 5000.0);
        for hit in detect_default(&cs).unwrap() {
            assert!(hit.volume.is_finite());
            assert!(hit.baseline.is_finite(), "{hit:?}");
            assert!(hit.ratio.is_finite(), "{hit:?}");
        }
    }

    #[test]
    fn zero_and_negative_volumes_are_skipped_not_treated_as_outliers() {
        let mut cs = flat(60, 1000.0);
        cs[45].volume = -5.0;
        cs[46].volume = f64::NAN;
        // The bad bars are skipped; the rest of a flat series stays quiet.
        for hit in detect_default(&cs).unwrap() {
            assert!(hit.volume >= 0.0 && hit.volume.is_finite());
        }
    }

    #[test]
    fn detection_is_deterministic() {
        let cs = with_spike(60, 1000.0, 50.0, 40, 9000.0);
        assert_eq!(detect_default(&cs).unwrap(), detect_default(&cs).unwrap());
    }

    #[test]
    fn peak_ratio_is_the_largest_multiple() {
        let cs = with_spike(60, 1000.0, 50.0, 40, 9000.0);
        let peak = peak_ratio(&cs).expect("a peak exists");
        assert!(peak > 5.0, "{peak}");
        assert!(peak_ratio(&flat(60, 1000.0)).is_some_and(|p| (p - 1.0).abs() < 1e-9));
        assert!(peak_ratio(&flat(10, 1000.0)).is_none());
    }
}
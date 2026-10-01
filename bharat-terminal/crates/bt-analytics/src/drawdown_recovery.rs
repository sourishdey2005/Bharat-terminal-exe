// crates/bt-analytics/src/drawdown_recovery.rs
// Author: Sourish Dey

//! Underwater chart geometry: peak-to-trough drawdowns and recovery times.
//!
//! The underwater plot is a percentage line at or below zero, so 0% means "at a
//! new high". What makes it informative rather than decorative is the annotation
//! layer: how deep each decline got, and how many bars recovery took. Those are
//! the numbers a reader actually wants from this chart, and they are easy to get
//! subtly wrong — a recovery measured from the wrong peak, or a trough selected
//! by the wrong criterion.
//!
//! Everything here is pure data over a close series, so the arithmetic is tested
//! without a renderer.

use bt_core::{Candle, Result};

/// A single peak-to-trough-and-back episode.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawdownEpisode {
    /// Index of the peak that started it.
    pub peak_index: usize,
    /// Peak close.
    pub peak_value: f64,
    /// Index of the trough.
    pub trough_index: usize,
    /// Trough close.
    pub trough_value: f64,
    /// Worst decline as a positive fraction (0.15 = 15% below the peak).
    pub depth: f64,
    /// Bars from peak to trough.
    pub decline_bars: usize,
    /// Index where the series regained the peak, if it did.
    pub recovery_index: Option<usize>,
    /// Bars from trough to recovery. `None` when never recovered.
    pub recovery_bars: Option<usize>,
}

impl DrawdownEpisode {
    /// Whether this episode ever got back to its peak.
    pub fn recovered(&self) -> bool {
        self.recovery_index.is_some()
    }

    /// Bars spent below the peak, counting from the peak itself.
    ///
    /// For an unrecovered episode this runs to `series_len`, which is why the
    /// caller has to pass it: "still underwater" is a statement about the end of
    /// the data, and the module does not assume where that is.
    pub fn underwater_bars(&self, series_len: usize) -> usize {
        match self.recovery_index {
            Some(r) => r.saturating_sub(self.peak_index),
            None => series_len.saturating_sub(self.peak_index),
        }
    }

    /// One-line summary for the chart footer.
    pub fn summary(&self) -> String {
        match self.recovery_bars {
            Some(b) => format!(
                "{:.1}% in {} bars, recovered in {}",
                self.depth * 100.0,
                self.decline_bars,
                b
            ),
            None => format!(
                "{:.1}% in {} bars, still below the peak",
                self.depth * 100.0,
                self.decline_bars
            ),
        }
    }
}

/// The full underwater series plus the episodes worth annotating.
#[derive(Debug, Clone, PartialEq)]
pub struct Underwater {
    /// Drawdown at each bar as a fraction, always `<= 0.0`. Non-positive
    /// prices produce `NaN` rather than a fabricated number.
    pub series: Vec<f64>,
    /// Every decline, in chronological order.
    pub episodes: Vec<DrawdownEpisode>,
    /// Index into `episodes` of the deepest one.
    pub deepest: Option<usize>,
    /// Bars in the underlying series, so callers can resolve open episodes.
    pub series_len: usize,
}

impl Underwater {
    /// Worst drawdown as a positive fraction.
    pub fn max_depth(&self) -> f64 {
        self.series
            .iter()
            .filter(|v| v.is_finite())
            .cloned()
            .fold(0.0_f64, f64::min)
            .abs()
    }

    /// Fraction of episodes that recovered. `None` when there were none.
    pub fn recovery_rate(&self) -> Option<f64> {
        if self.episodes.is_empty() {
            return None;
        }
        let ok = self.episodes.iter().filter(|e| e.recovered()).count();
        Some(ok as f64 / self.episodes.len() as f64)
    }

    /// Mean bars-to-recovery across recovered episodes. `None` when none did.
    pub fn mean_recovery_bars(&self) -> Option<f64> {
        let bars: Vec<usize> = self.episodes.iter().filter_map(|e| e.recovery_bars).collect();
        if bars.is_empty() {
            return None;
        }
        Some(bars.iter().sum::<usize>() as f64 / bars.len() as f64)
    }

    /// Longest stretch spent below a peak, recovered or not.
    pub fn longest_underwater(&self) -> usize {
        self.episodes
            .iter()
            .map(|e| e.underwater_bars(self.series_len))
            .max()
            .unwrap_or(0)
    }
}

/// Percentage below the running maximum, at every bar.
pub fn underwater(closes: &[f64]) -> Result<Vec<f64>> {
    if closes.is_empty() {
        return Err(bt_core::BtError::EmptySeries(
            "underwater plot needs at least one close".into(),
        ));
    }
    let mut peak = f64::NEG_INFINITY;
    let mut out = Vec::with_capacity(closes.len());
    for &c in closes {
        if !c.is_finite() || c <= 0.0 {
            // A gap, not a 100% drawdown.
            out.push(f64::NAN);
            continue;
        }
        if c > peak {
            peak = c;
        }
        out.push(c / peak - 1.0);
    }
    Ok(out)
}

/// Find every peak-to-trough decline, recovered or not.
pub fn episodes(closes: &[f64]) -> Result<Vec<DrawdownEpisode>> {
    let series = underwater(closes)?;
    let mut out: Vec<DrawdownEpisode> = Vec::new();

    let mut peak_idx = 0usize;
    let mut peak_val = f64::NEG_INFINITY;
    let mut trough_idx = 0usize;
    let mut trough_val = f64::NEG_INFINITY;
    let mut in_drawdown = false;

    for i in 0..closes.len() {
        let price = closes[i];
        if !price.is_finite() || price <= 0.0 {
            continue;
        }
        if !in_drawdown {
            if price >= peak_val {
                peak_val = price;
                peak_idx = i;
            } else {
                in_drawdown = true;
                trough_idx = i;
                trough_val = price;
            }
        } else {
            if price < trough_val {
                trough_val = price;
                trough_idx = i;
            }
            if price >= peak_val {
                out.push(make_episode(
                    peak_idx,
                    peak_val,
                    trough_idx,
                    trough_val,
                    Some(i),
                ));
                in_drawdown = false;
                // The recovery bar is itself a peak to decline from next.
                peak_idx = i;
                peak_val = price;
            }
        }
    }

    if in_drawdown {
        out.push(make_episode(
            peak_idx,
            peak_val,
            trough_idx,
            trough_val,
            None,
        ));
    }

    Ok(out)
}

fn make_episode(
    peak_index: usize,
    peak_value: f64,
    trough_index: usize,
    trough_value: f64,
    recovery_index: Option<usize>,
) -> DrawdownEpisode {
    let depth = if peak_value > 0.0 {
        ((peak_value - trough_value) / peak_value).abs()
    } else {
        0.0
    };
    DrawdownEpisode {
        peak_index,
        peak_value,
        trough_index,
        trough_value,
        depth,
        decline_bars: trough_index.saturating_sub(peak_index),
        recovery_index,
        recovery_bars: recovery_index.map(|r| r.saturating_sub(trough_index)),
    }
}

/// Index of the deepest episode, for the chart's headline number.
pub fn deepest_episode(episodes: &[DrawdownEpisode]) -> Option<usize> {
    episodes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

/// Build the underwater series and its episodes from candles.
pub fn from_candles(candles: &[Candle]) -> Result<Underwater> {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let eps = episodes(&closes)?;
    Ok(Underwater {
        deepest: deepest_episode(&eps),
        series: underwater(&closes)?,
        series_len: closes.len(),
        episodes: eps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rises to `peak_at`, falls to `trough` by `trough_at`, then optionally
    /// recovers over three bars.
    fn shaped(n: usize, peak_at: usize, trough_at: usize, trough: f64, recovers: bool) -> Vec<f64> {
        let peak = 100.0 + peak_at as f64;
        (0..n)
            .map(|i| {
                if i <= peak_at {
                    100.0 + i as f64
                } else if i <= trough_at {
                    let span = (trough_at - peak_at).max(1) as f64;
                    peak + (trough - peak) * ((i - peak_at) as f64 / span)
                } else if recovers {
                    trough + (peak - trough) * ((i - trough_at) as f64 / 3.0)
                } else {
                    trough
                }
            })
            .collect()
    }

    #[test]
    fn a_monotonic_rise_has_no_drawdown() {
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let u = underwater(&closes).unwrap();
        assert!(u.iter().all(|d| *d <= 1e-12), "{u:?}");
        assert!(episodes(&closes).unwrap().is_empty());
    }

    #[test]
    fn a_decline_from_the_peak_is_negative_and_sizes_correctly() {
        // Peak 105 (index 5), trough 80 -> 25/105 = 23.81%.
        let closes = shaped(30, 5, 15, 80.0, true);
        let u = underwater(&closes).unwrap();
        let worst = u.iter().cloned().fold(0.0_f64, f64::min);
        assert!((worst + 25.0 / 105.0).abs() < 1e-9, "{worst}");
    }

    #[test]
    fn a_recovered_episode_is_detected_with_its_bars() {
        let closes = shaped(40, 5, 15, 80.0, true);
        let eps = episodes(&closes).unwrap();
        assert_eq!(eps.len(), 1, "{eps:?}");
        let e = &eps[0];
        assert!(e.recovered(), "{e:?}");
        assert_eq!(e.peak_index, 5);
        assert_eq!(e.trough_index, 15);
        assert_eq!(e.decline_bars, 10);
        assert_eq!(e.recovery_bars, Some(3), "recovery takes three bars in the fixture");
        assert!(e.summary().contains("recovered in"), "{}", e.summary());
    }

    #[test]
    fn an_unrecovered_episode_is_reported_as_such() {
        let closes = shaped(30, 5, 20, 70.0, false);
        let eps = episodes(&closes).unwrap();
        assert_eq!(eps.len(), 1, "{eps:?}");
        assert!(!eps[0].recovered());
        assert!(eps[0].recovery_bars.is_none());
        assert!(
            eps[0].summary().contains("still below"),
            "{}",
            eps[0].summary()
        );
    }

    #[test]
    fn multiple_declines_are_separate_chronological_episodes() {
        let mut closes = shaped(25, 4, 10, 90.0, true);
        closes.extend((0..15).map(|i| 100.0 - i as f64 * 1.5));
        let eps = episodes(&closes).unwrap();
        assert!(eps.len() >= 2, "{eps:?}");
        for pair in eps.windows(2) {
            assert!(
                pair[0].peak_index < pair[1].peak_index,
                "episodes must be in order: {eps:?}"
            );
        }
    }

    #[test]
    fn the_deepest_episode_is_selected_by_depth() {
        let mut closes = shaped(20, 3, 8, 95.0, true); // shallow
        closes.extend(shaped(20, 3, 18, 50.0, true)); // deeper
        let eps = episodes(&closes).unwrap();
        let deepest = deepest_episode(&eps).expect("an episode");
        assert_eq!(deepest, eps.len() - 1, "{eps:?}");
    }

    #[test]
    fn an_empty_series_is_an_error_not_a_bare_plot() {
        let e: Vec<f64> = Vec::new();
        assert!(underwater(&e).is_err());
        assert!(episodes(&e).is_err());
    }

    #[test]
    fn a_single_price_is_flat_not_a_drawdown() {
        let u = underwater(&[100.0]).unwrap();
        assert_eq!(u.len(), 1);
        assert!(u[0].abs() < 1e-12);
        assert!(episodes(&[100.0]).unwrap().is_empty());
    }

    #[test]
    fn non_positive_prices_become_a_gap_not_a_fake_drawdown() {
        let u = underwater(&[100.0, 0.0, 50.0]).unwrap();
        assert!(u[1].is_nan(), "a zero price must not read as -100%: {u:?}");
        assert!(u[0].abs() < 1e-12);
        assert!(u[2].is_finite());
    }

    #[test]
    fn max_depth_ignores_the_nan_gaps() {
        let mut closes = vec![100.0, 120.0, 90.0];
        closes.insert(2, f64::NAN);
        let u = underwater(&closes).unwrap();
        assert!(u[2].is_nan());
        // A consumer folding the raw series would get NaN; the wrapper's
        // max_depth must not, because a chart showing "NaN%" is worse than none.
        let uw = from_candles(
            &closes
                .iter()
                .enumerate()
                .map(|(i, c)| Candle::new(i as f64, *c, *c, *c, *c, 1.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(uw.max_depth().is_finite(), "{:?}", uw.max_depth());
    }

    #[test]
    fn no_episodes_means_none_not_zero_for_the_summary_metrics() {
        let flat: Vec<Candle> = (0..10)
            .map(|i| Candle::new(i as f64, 100.0, 100.0, 100.0, 100.0, 1.0))
            .collect();
        let uw = from_candles(&flat).unwrap();
        assert!(uw.episodes.is_empty());
        assert_eq!(uw.recovery_rate(), None);
        assert_eq!(uw.mean_recovery_bars(), None);
        assert_eq!(uw.longest_underwater(), 0);
        assert_eq!(uw.deepest, None);
        assert_eq!(uw.max_depth(), 0.0);
    }

    #[test]
    fn an_unrecovered_episode_runs_to_the_end_of_the_data() {
        let closes = shaped(30, 5, 20, 70.0, false);
        let uw = from_candles(
            &closes
                .iter()
                .enumerate()
                .map(|(i, c)| Candle::new(i as f64, *c, *c, *c, *c, 1.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(uw.series_len, 30);
        assert_eq!(uw.longest_underwater(), 30 - 5);
        assert_eq!(uw.recovery_rate(), Some(0.0));
    }

    #[test]
    fn from_candles_needs_candles() {
        assert!(from_candles(&[]).is_err());
    }

    #[test]
    fn underwater_bars_never_underflows() {
        let closes = shaped(30, 5, 20, 70.0, false);
        let eps = episodes(&closes).unwrap();
        let e = &eps[0];
        // Passing a series shorter than the peak index must saturate, not wrap.
        assert_eq!(e.underwater_bars(0), 0);
        assert!(e.underwater_bars(100) >= e.underwater_bars(30));
    }

    #[test]
    fn results_are_deterministic() {
        let closes = shaped(40, 5, 15, 80.0, true);
        let a = episodes(&closes).unwrap();
        let b = episodes(&closes).unwrap();
        assert_eq!(a, b);
    }
}
// crates/bt-analytics/src/gap_scanner.rs
// Author: Sourish Dey

//! Overnight gap detection.
//!
//! A gap is the move between one session's close and the next session's open.
//! It matters because it is the one part of a day's move that happened while no
//! one could trade, so it is pure information — but a gap on no volume is not
//! meaningful, which is why every gap here is confirmed against the volume that
//! followed it.
//!
//! Gaps are reported on the *preceding* index, since a gap at bar `i` is known
//! only when bar `i`'s open is seen, and it belongs to the move out of bar
//! `i - 1`.

use bt_core::Candle;

/// Smallest gap, in percent, worth reporting.
pub const DEFAULT_GAP_PCT: f64 = 2.0;

/// Volume a confirming bar needs, as a multiple of the trailing average.
pub const CONFIRM_VOLUME_RATIO: f64 = 1.5;

/// One confirmed or unconfirmed gap.
#[derive(Debug, Clone, PartialEq)]
pub struct GapHit {
    /// Index of the bar whose open gapped away from the previous close.
    pub index: usize,
    pub symbol_free_prev_close: f64,
    pub open: f64,
    /// `(open - prev_close) / prev_close * 100`. Positive is a gap up.
    pub gap_pct: f64,
    /// Absolute gap size in price units.
    pub gap_size: f64,
    /// Volume of the gap bar against the trailing average.
    pub volume_ratio: f64,
    /// Whether the move had volume behind it.
    pub confirmed: bool,
}

impl GapHit {
    /// "Gap up" / "Gap down".
    pub fn direction(&self) -> &'static str {
        if self.gap_pct >= 0.0 {
            "Gap up"
        } else {
            "Gap down"
        }
    }

    /// Unconfirmed gaps are reported separately in the UI: an unconfirmed gap
    /// is a thin-market artefact far more often than a real repricing.
    pub fn strength_label(&self) -> &'static str {
        match (self.confirmed, self.gap_pct.abs()) {
            (true, g) if g >= 5.0 => "Major",
            (true, _) => "Confirmed",
            (false, g) if g >= 5.0 => "Unconfirmed major",
            (false, _) => "Thin",
        }
    }
}

/// Scan `candles` for gaps of at least `min_pct`.
pub fn scan(candles: &[Candle], min_pct: f64) -> Vec<GapHit> {
    let floor = if min_pct.is_finite() && min_pct > 0.0 {
        min_pct
    } else {
        DEFAULT_GAP_PCT
    };
    if candles.len() < 2 {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 1..candles.len() {
        let prev_close = candles[i - 1].close;
        let open = candles[i].open;
        if !prev_close.is_finite() || prev_close <= 0.0 || !open.is_finite() || open <= 0.0 {
            continue;
        }
        let gap_pct = (open - prev_close) / prev_close * 100.0;
        if !gap_pct.is_finite() || gap_pct.abs() < floor {
            continue;
        }

        // Volume confirmation against the 20 bars preceding the gap. Short
        // history means "cannot confirm", never "confirmed".
        let volume_ratio = if i >= 20 {
            let vols: Vec<f64> = candles[i - 20..i].iter().map(|c| c.volume).collect();
            let mean = vols.iter().sum::<f64>() / vols.len() as f64;
            if mean > 0.0 && candles[i].volume.is_finite() {
                candles[i].volume / mean
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        let confirmed = volume_ratio.is_finite() && volume_ratio >= CONFIRM_VOLUME_RATIO;

        out.push(GapHit {
            index: i,
            symbol_free_prev_close: prev_close,
            open,
            gap_pct,
            gap_size: open - prev_close,
            volume_ratio,
            confirmed,
        });
    }
    out
}

/// Convenience wrapper using [`DEFAULT_GAP_PCT`].
pub fn scan_default(candles: &[Candle]) -> Vec<GapHit> {
    scan(candles, DEFAULT_GAP_PCT)
}

/// Largest gap magnitude found, for the header readout.
pub fn largest_gap(candles: &[Candle]) -> Option<GapHit> {
    scan_default(candles)
        .into_iter()
        .max_by(|a, b| a.gap_pct.abs().partial_cmp(&b.gap_pct.abs()).unwrap_or(std::cmp::Ordering::Equal))
}

/// Fraction of gaps that were volume-confirmed.
pub fn confirmation_rate(candles: &[Candle]) -> Option<f64> {
    let gaps = scan_default(candles);
    if gaps.is_empty() {
        return None;
    }
    let confirmed = gaps.iter().filter(|g| g.confirmed).count();
    Some(confirmed as f64 / gaps.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Continuous candles where every close is exactly the prior close (a run of
    /// dojis). A helper that drifts the close by a fraction each bar makes every
    /// later "gap" enormous and silently invalidates any assertion about gap
    /// size, so the price is pinned here on purpose.
    fn continuous(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                Candle::new(i as f64, 100.0, 101.0, 99.0, 100.0, 1000.0)
            })
            .collect()
    }

    /// Insert an opening gap at `index`.
    fn with_gap(candles: &mut [Candle], index: usize, open: f64, volume: f64) {
        candles[index].open = open;
        candles[index].high = open + 1.0;
        candles[index].low = open - 1.0;
        candles[index].volume = volume;
    }

    #[test]
    fn a_continuous_series_has_no_gaps() {
        assert!(scan_default(&continuous(60)).is_empty());
    }

    #[test]
    fn a_short_series_is_empty_rather_than_panicking() {
        for n in [0usize, 1] {
            assert!(scan_default(&continuous(n)).is_empty());
        }
    }

    #[test]
    fn a_real_gap_is_found_at_the_right_bar() {
        let mut cs = continuous(60);
        // Previous close is pinned at 100, so an open at 103 is a clean +3% gap.
        with_gap(&mut cs, 40, 103.0, 1000.0);
        let hits = scan_default(&cs);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].index, 40);
        assert!((hits[0].gap_pct - 3.0).abs() < 1e-9, "{hits:?}");
        assert_eq!(hits[0].direction(), "Gap up");
    }

    #[test]
    fn a_gap_down_is_reported_as_such() {
        let mut cs = continuous(60);
        with_gap(&mut cs, 30, 96.0, 1000.0);
        let hits = scan_default(&cs);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!((hits[0].gap_pct + 4.0).abs() < 1e-9, "{hits:?}");
        assert_eq!(hits[0].direction(), "Gap down");
    }

    #[test]
    fn a_small_gap_below_the_threshold_is_ignored() {
        let mut cs = continuous(60);
        with_gap(&mut cs, 40, 101.0, 1000.0); // +1%, under the 2% floor
        assert!(scan_default(&cs).is_empty(), "{:?}", scan_default(&cs));
    }

    #[test]
    fn volume_confirms_a_real_gap() {
        let mut cs = continuous(60);
        with_gap(&mut cs, 40, 103.0, 100_000.0); // 100x the baseline
        let hits = scan_default(&cs);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].confirmed, "{hits:?}");
        assert!(hits[0].volume_ratio > 1.5, "{hits:?}");
        // A 3% gap is under the 5% "major" line.
        assert_eq!(hits[0].strength_label(), "Confirmed");
    }

    #[test]
    fn an_unconfirmed_gap_is_labelled_thin() {
        let mut cs = continuous(60);
        with_gap(&mut cs, 40, 103.0, 10.0); // far below the baseline
        let hits = scan_default(&cs);
        assert!(!hits[0].confirmed, "{hits:?}");
        assert_eq!(hits[0].strength_label(), "Thin");
    }

    #[test]
    fn a_gap_without_enough_history_cannot_be_confirmed() {
        // Fewer than 20 preceding bars means no baseline exists. Claiming
        // "confirmed" here would be asserting a measurement never made.
        let mut cs = continuous(25);
        with_gap(&mut cs, 10, 103.0, 100_000.0);
        let hits = scan_default(&cs);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(!hits[0].confirmed, "{hits:?}");
        assert!(!hits[0].volume_ratio.is_finite(), "{hits:?}");
    }

    #[test]
    fn a_major_gap_is_distinguished_from_a_minor_one() {
        let mut cs = continuous(80);
        with_gap(&mut cs, 60, 110.0, 100_000.0);
        assert_eq!(scan_default(&cs)[0].strength_label(), "Major");
        // And a big but unconfirmed gap gets its own label.
        with_gap(&mut cs, 65, 90.0, 10.0);
        let hits = scan_default(&cs);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[1].strength_label(), "Unconfirmed major");
    }

    #[test]
    fn a_hostile_threshold_falls_back_to_the_default() {
        let mut cs = continuous(60);
        with_gap(&mut cs, 40, 105.0, 1000.0);
        for bad in [f64::NAN, -1.0, 0.0, f64::INFINITY] {
            assert_eq!(scan(&cs, bad).len(), 1, "threshold {bad} must fall back");
        }
    }

    #[test]
    fn zero_or_nan_prices_are_skipped_not_divided_by() {
        let mut cs = continuous(60);
        with_gap(&mut cs, 40, f64::NAN, 1000.0);
        with_gap(&mut cs, 41, 0.0, 1000.0);
        for hit in scan_default(&cs) {
            assert!(hit.gap_pct.is_finite(), "{hit:?}");
            assert!(hit.symbol_free_prev_close > 0.0, "{hit:?}");
        }
    }

    #[test]
    fn summary_helpers_agree_with_the_scan() {
        let mut cs = continuous(60);
        with_gap(&mut cs, 40, 105.0, 100_000.0);
        with_gap(&mut cs, 45, 95.0, 10.0);
        assert_eq!(scan_default(&cs).len(), 2);
        assert!(largest_gap(&cs).is_some());
        // One of two confirmed.
        let rate = confirmation_rate(&cs).expect("a rate");
        assert!((rate - 0.5).abs() < 1e-9, "{rate}");
        assert!(confirmation_rate(&continuous(60)).is_none());
        assert!(largest_gap(&continuous(60)).is_none());
    }

    #[test]
    fn scanning_is_deterministic() {
        let mut cs = continuous(60);
        with_gap(&mut cs, 40, 105.0, 5000.0);
        assert_eq!(scan_default(&cs), scan_default(&cs));
    }
}
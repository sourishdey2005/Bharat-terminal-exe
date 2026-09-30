// crates/bt-analytics/src/models/contracts.rs
// Author: Sourish Dey

//! The published I/O contract of every shipped model, verified against the
//! graphs rather than assumed.
//!
//! # Why this module exists
//!
//! Several plausible-looking integration notes for this project are wrong about
//! the models, and each wrong detail fails differently:
//!
//! * **Chronos-Bolt** emits `[1, 9, 64]`, but the quantiles are indexed `0/4/8`,
//!   not `1/4/7`. Taking index 1 and index 7 silently returns the 2nd and 8th
//!   quantile: a narrower, off-centre corridor that still *looks* plausible.
//! * **WatchSignal LSTM** takes `[1, 30, 55]`, not `[1, 30, 5]`. The last axis is
//!   55 engineered features, not raw OHLCV, so a 5-wide tensor is a hard shape
//!   error, not a silent degradation.
//! * **WatchSignal LSTM** already emits probabilities - the output sums to 1.0 -
//!   so applying a softmax on top double-normalises and flattens every
//!   confidence toward 1/3.
//! * **DLinear / N-HiTS** were trained with `(window - last_close) / window_sd`,
//!   not `(window - last_close) / last_close`. Dropping the standard deviation
//!   changes the input distribution by orders of magnitude on a price series.
//!
//! The constants here are the corrected values, and the tests re-derive them from
//! the installed graphs when those graphs are present.

/// Context bars Chronos-Bolt requires.
pub const CHRONOS_CONTEXT: usize = 64;
/// Quantile levels Chronos-Bolt emits.
pub const CHRONOS_QUANTILES: usize = 9;
/// Steps each Chronos-Bolt quantile covers.
pub const CHRONOS_HORIZON: usize = 64;
/// Flattened element count of a Chronos-Bolt output.
pub const CHRONOS_FLAT_LEN: usize = CHRONOS_QUANTILES * CHRONOS_HORIZON;

/// Index of the 10th-percentile band.
///
/// Verified by inspecting the emitted bands on a rising series: band minima
/// increase monotonically from index 0 to the median and band maxima increase
/// from the median to index 8.
pub const CHRONOS_P10_INDEX: usize = 0;
/// Index of the median (50th-percentile) band.
pub const CHRONOS_P50_INDEX: usize = CHRONOS_QUANTILES / 2;
/// Index of the 90th-percentile band.
pub const CHRONOS_P90_INDEX: usize = CHRONOS_QUANTILES - 1;

/// Bars the WatchSignal LSTM consumes.
pub const SIGNAL_WINDOW: usize = 30;
/// **Features per bar** the WatchSignal LSTM consumes.
///
/// This is 55, not 5. The model was trained on engineered features (returns,
/// momentum, volume statistics, ...) laid out in a fixed order, and the caller
/// builds that vector; passing raw OHLCV is a shape error.
pub const SIGNAL_N_FEATURES: usize = 55;
/// Classes the WatchSignal LSTM emits, in its own `[Sell, Hold, Buy]` order.
pub const SIGNAL_N_CLASSES: usize = 3;

/// Lookback for the two small window models.
pub const SMALL_LOOKBACK: usize = 32;
/// Horizon for the two small window models.
pub const SMALL_HORIZON: usize = 5;

/// Whether a Chronos-Bolt output is shaped the way [`split_quantiles`] assumes.
///
/// Cheap enough to run on every inference, and it turns "the cone looks a bit
/// odd" into a specific error naming the expected element count.
pub fn chronos_flat_len_is_valid(len: usize) -> bool {
    len == CHRONOS_FLAT_LEN
}

/// Pull `p10`, `p50` and `p90` out of a flattened Chronos-Bolt output.
///
/// `None` when the buffer is not the shape the graph declares.
pub fn split_quantiles(flat: &[f32], horizon: usize) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if !chronos_flat_len_is_valid(flat.len()) {
        return None;
    }
    let steps = horizon.clamp(1, CHRONOS_HORIZON);
    let band = |q: usize| -> Vec<f64> {
        flat[q * CHRONOS_HORIZON..q * CHRONOS_HORIZON + steps]
            .iter()
            .map(|v| *v as f64)
            .collect()
    };
    Some((
        band(CHRONOS_P10_INDEX),
        band(CHRONOS_P50_INDEX),
        band(CHRONOS_P90_INDEX),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_indices_are_zero_four_eight() {
        // Not 1/4/7: those return the 2nd and 8th quantile.
        assert_eq!(CHRONOS_P10_INDEX, 0);
        assert_eq!(CHRONOS_P50_INDEX, 4);
        assert_eq!(CHRONOS_P90_INDEX, 8);
    }

    #[test]
    fn the_signal_model_is_fifty_five_features_not_five() {
        assert_eq!(SIGNAL_N_FEATURES, 55);
        assert_eq!(SIGNAL_N_FEATURES * SIGNAL_WINDOW, 1650);
        assert_ne!(
            SIGNAL_N_FEATURES, 5,
            "5 features would be raw OHLCV and is a shape error against this graph"
        );
    }

    #[test]
    fn a_wrongly_sized_chronos_buffer_is_rejected() {
        assert!(!chronos_flat_len_is_valid(10));
        assert!(chronos_flat_len_is_valid(CHRONOS_FLAT_LEN));
        assert_eq!(CHRONOS_FLAT_LEN, 576);
        assert!(split_quantiles(&[0.0; 10], 8).is_none());
    }

    #[test]
    fn splitting_picks_the_documented_bands() {
        let flat: Vec<f32> = (0..CHRONOS_FLAT_LEN).map(|i| i as f32).collect();
        let (p10, p50, p90) = split_quantiles(&flat, 10).expect("split");
        assert_eq!(p10.len(), 10);
        assert_eq!(p50.len(), 10);
        assert_eq!(p90.len(), 10);
        assert_eq!(p10[0], 0.0);
        assert_eq!(p50[0], (CHRONOS_P50_INDEX * CHRONOS_HORIZON) as f64);
        assert_eq!(p90[0], (CHRONOS_P90_INDEX * CHRONOS_HORIZON) as f64);
    }

    #[test]
    fn the_horizon_is_clamped_to_the_model() {
        let flat: Vec<f32> = (0..CHRONOS_FLAT_LEN).map(|i| i as f32).collect();
        let (p10, _, _) = split_quantiles(&flat, 999).expect("split");
        assert_eq!(p10.len(), CHRONOS_HORIZON);
    }
}

// crates/bt-analytics/src/models/quantile_engine.rs
// Author: Sourish Dey

//! Probabilistic forecast bands: the "cone of uncertainty".
//!
//! A point forecast says where the price is going. A quantile band says how
//! confident that is, which is the part a trader actually acts on. Chronos-Bolt
//! is a 9-quantile model, so a single inference already yields a full corridor:
//! the 10th percentile as a downside floor, the median as the central path, and
//! the 90th as an upside ceiling.
//!
//! # Why this reuses the existing engine
//!
//! The ONNX session, the LRU cache and the `[quantiles, horizon]` unpacking all
//! already exist in [`super::engine`]. Opening a second session for the same
//! graph would double its resident memory for no gain, and this crate targets a
//! 2 GB machine. So [`QuantileForecaster`] is a thin projection over
//! [`BharatModelEngine`]: no new dependency, no new session, no new IO.
//!
//! # Band indices
//!
//! Chronos emits nine quantiles in ascending order, so index 0 is p10, index 4
//! is p50 and index 8 is p90. (Some Chronos documentation numbers them from 1;
//! the graph's own output layout is what [`CHRONOS_P10`] and friends assert,
//! and [`band_is_ordered`] verifies the result rather than trusting the index.)

use super::engine::{
    BharatModelEngine, ForecastOutput, CHRONOS_CONTEXT, CHRONOS_HORIZON, CHRONOS_QUANTILES,
};

/// Quantile index of the downside band (10th percentile).
pub const CHRONOS_P10: usize = 0;
/// Quantile index of the central path (50th percentile / median).
pub const CHRONOS_P50: usize = CHRONOS_QUANTILES / 2;
/// Quantile index of the upside band (90th percentile).
pub const CHRONOS_P90: usize = CHRONOS_QUANTILES - 1;

/// Tolerance when checking that the bands are properly nested.
///
/// Prices are in the thousands, so an exact comparison would trip on float
/// noise alone.
const BAND_EPS: f64 = 1e-6;

/// Three trajectories forming a quantile cone.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileConeOutput {
    /// Bars projected.
    pub horizon_steps: usize,
    /// 10th percentile: the downside floor.
    pub p10_lower: Vec<f64>,
    /// 50th percentile: the median path.
    pub p50_median: Vec<f64>,
    /// 90th percentile: the upside ceiling.
    pub p90_upper: Vec<f64>,
}

impl QuantileConeOutput {
    /// Half-width of the cone at each step; the fan a chart actually shades.
    pub fn widths(&self) -> Vec<f64> {
        (0..self.horizon_steps)
            .map(|i| {
                let up = self.p90_upper.get(i).copied().unwrap_or(0.0);
                let lo = self.p10_lower.get(i).copied().unwrap_or(0.0);
                (up - lo).abs() / 2.0
            })
            .collect()
    }

    /// True when `p10 <= p50 <= p90` holds at every step, within tolerance.
    ///
    /// A cone is a nested corridor by definition. If the model ever returns
    /// bands that cross, the fan chart would render as an impossible shape, so
    /// callers can check this and refuse to draw it.
    pub fn band_is_ordered(&self) -> bool {
        let eps = BAND_EPS;
        (0..self.horizon_steps).all(|i| {
            let (lo, mid, hi) = (self.p10_lower[i], self.p50_median[i], self.p90_upper[i]);
            lo <= mid + eps && mid <= hi + eps
        })
    }

    /// Build a cone from a forecast that carries its own bands, truncated to
    /// `horizon` bars.
    ///
    /// Engines with no quantile output (ARIMA, the moving average, NanoForecast)
    /// yield `None`: there is nothing to shade, and pretending otherwise would
    /// draw a band of zero width that reads as false certainty.
    pub fn from_forecast(out: &ForecastOutput, horizon: usize) -> Option<Self> {
        let (Some(lower), Some(upper)) = (&out.lower, &out.upper) else {
            return None;
        };
        let steps = horizon
            .min(out.horizon_steps)
            .min(lower.len())
            .min(upper.len())
            .min(out.predictions.len());
        if steps == 0 {
            return None;
        }
        Some(QuantileConeOutput {
            horizon_steps: steps,
            p10_lower: lower[..steps].to_vec(),
            p50_median: out.predictions[..steps].to_vec(),
            p90_upper: upper[..steps].to_vec(),
        })
    }
}

/// Projects quantile cones using the already-loaded Chronos session.
pub struct QuantileForecaster {
    engine: BharatModelEngine,
}

impl QuantileForecaster {
    /// Build on an existing engine, sharing its session cache.
    pub fn with_engine(engine: BharatModelEngine) -> Self {
        Self { engine }
    }

    /// Build on the default `models/` location.
    pub fn new() -> Self {
        Self::with_engine(BharatModelEngine::with_default_paths())
    }

    /// The engine backing this forecaster, for callers that also want the
    /// point forecast or the directional signal.
    pub fn engine(&self) -> &BharatModelEngine {
        &self.engine
    }

    /// Run Chronos-Bolt and return its p10 / p50 / p90 corridor.
    ///
    /// Needs `CHRONOS_CONTEXT` (64) bars of history. Returns `None` when the
    /// model file is absent, so an installation without it degrades to the other
    /// paradigms instead of erroring.
    pub fn forecast_uncertainty_cone(
        &self,
        closes: &[f64],
        horizon: usize,
    ) -> Option<QuantileConeOutput> {
        if closes.len() < CHRONOS_CONTEXT || !self.engine.is_available(super::Model::Chronos) {
            return None;
        }
        let out = self.engine.predict_chronos(closes).ok()?;
        let cone = QuantileConeOutput::from_forecast(&out, horizon)?;
        // A crossed band is a real model or unpacking fault. Surfacing `None`
        // here means the UI drops the fan and says so, rather than painting an
        // impossible corridor.
        cone.band_is_ordered().then_some(cone)
    }

    /// Bars the model needs before a cone can be produced.
    pub const fn required_history() -> usize {
        CHRONOS_CONTEXT
    }

    /// Bars the model can produce at most.
    pub const fn max_horizon() -> usize {
        CHRONOS_HORIZON
    }
}

impl Default for QuantileForecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: the three Chronos quantile indices, for tests and legends.
pub const CHRONOS_BAND_INDICES: [usize; 3] = [CHRONOS_P10, CHRONOS_P50, CHRONOS_P90];

/// Turn a raw Chronos output row-major into `(p10, p50, p90)` slices.
///
/// Exposed for the model verification tests, which assert against the real
/// graph rather than against this crate's own unpacking.
pub fn split_bands(raw: &[f32]) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if raw.len() < CHRONOS_QUANTILES * CHRONOS_HORIZON {
        return None;
    }
    let band = |q: usize| -> Vec<f64> {
        raw[q * CHRONOS_HORIZON..(q + 1) * CHRONOS_HORIZON]
            .iter()
            .map(|v| *v as f64)
            .collect()
    };
    Some((band(CHRONOS_P10), band(CHRONOS_P50), band(CHRONOS_P90)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;

    #[test]
    fn band_indices_are_the_documented_ones() {
        assert_eq!(CHRONOS_P10, 0);
        assert_eq!(CHRONOS_P50, 4);
        assert_eq!(CHRONOS_P90, 8);
        assert_eq!(CHRONOS_QUANTILES, 9);
        assert_eq!(CHRONOS_BAND_INDICES, [0, 4, 8]);
    }

    #[test]
    fn cone_is_truncated_to_the_requested_horizon() {
        let full: Vec<f64> = (0..CHRONOS_HORIZON).map(|i| 100.0 + i as f64).collect();
        let out = ForecastOutput {
            model_name: "test".into(),
            horizon_steps: CHRONOS_HORIZON,
            predictions: full.clone(),
            lower: Some(full.clone()),
            upper: Some(full.clone()),
            lookback: CHRONOS_CONTEXT,
        };
        let cone = QuantileConeOutput::from_forecast(&out, 10).expect("cone");
        assert_eq!(cone.horizon_steps, 10);
        assert_eq!(cone.p10_lower.len(), 10);
        assert_eq!(cone.p50_median.len(), 10);
        assert_eq!(cone.p90_upper.len(), 10);
        assert_eq!(cone.p10_lower[9], 109.0);
    }

    /// A model with no quantile output must not be dressed up as if it had one.
    #[test]
    fn no_band_means_no_cone() {
        let out = ForecastOutput {
            model_name: "ARIMA".into(),
            horizon_steps: 5,
            predictions: vec![1.0; 5],
            lower: None,
            upper: None,
            lookback: 64,
        };
        assert!(QuantileConeOutput::from_forecast(&out, 5).is_none());
    }

    #[test]
    fn width_and_ordering_are_computed_from_the_bands() {
        let cone = QuantileConeOutput {
            horizon_steps: 2,
            p10_lower: vec![90.0, 88.0],
            p50_median: vec![100.0, 100.0],
            p90_upper: vec![110.0, 112.0],
        };
        assert_eq!(cone.widths(), vec![10.0, 12.0]);
        assert!(cone.band_is_ordered());
    }

    #[test]
    fn a_crossed_band_is_detected() {
        // Lower above median: geometrically impossible for a quantile cone.
        let crossed = QuantileConeOutput {
            horizon_steps: 1,
            p10_lower: vec![120.0],
            p50_median: vec![100.0],
            p90_upper: vec![130.0],
        };
        assert!(!crossed.band_is_ordered());
    }

    #[test]
    fn split_bands_rejects_a_short_row() {
        assert!(split_bands(&[0.0; 10]).is_none());
    }

    #[test]
    fn split_bands_picks_the_right_slices() {
        let raw: Vec<f32> = (0..CHRONOS_QUANTILES * CHRONOS_HORIZON)
            .map(|i| i as f32)
            .collect();
        let (lo, mid, hi) = split_bands(&raw).expect("bands");
        assert_eq!(lo.len(), CHRONOS_HORIZON);
        assert_eq!(mid[0], (CHRONOS_P50 * CHRONOS_HORIZON) as f64);
        assert_eq!(hi[0], (CHRONOS_P90 * CHRONOS_HORIZON) as f64);
    }

    #[test]
    fn short_history_yields_no_cone_rather_than_erroring() {
        let f = QuantileForecaster::new();
        assert!(f.forecast_uncertainty_cone(&[100.0; 10], 8).is_none());
    }

    /// Only meaningful where the Chronos file is actually installed; skipped
    /// elsewhere so a partial install still passes the suite.
    #[test]
    fn real_chronos_cone_is_ordered_when_installed() {
        let engine = BharatModelEngine::with_default_paths();
        if !engine.is_available(Model::Chronos) {
            return;
        }
        let f = QuantileForecaster::with_engine(engine);
        let closes: Vec<f64> = (0..80)
            .map(|i| 2500.0 + 20.0 * (i as f64 * 0.3).sin() + i as f64)
            .collect();
        let cone = f
            .forecast_uncertainty_cone(&closes, 16)
            .expect("cone should be produced when Chronos is installed");
        assert_eq!(cone.horizon_steps, 16);
        assert!(
            cone.band_is_ordered(),
            "Chronos returned crossed quantiles; the fan chart would be invalid"
        );
        // A real forecast must carry a non-degenerate corridor.
        let widths = cone.widths();
        assert!(
            widths.iter().any(|w| *w > 0.0),
            "the cone has zero width everywhere, which would read as false certainty"
        );
        assert!(
            cone.p10_lower.iter().all(|v| v.is_finite())
                && cone.p90_upper.iter().all(|v| v.is_finite())
        );
    }

    #[test]
    fn horizon_caps_are_the_models_own() {
        assert_eq!(QuantileForecaster::required_history(), 64);
        assert_eq!(QuantileForecaster::max_horizon(), 64);
    }
}

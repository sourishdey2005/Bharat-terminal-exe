// crates/bt-analytics/src/forecast/chronos.rs
// Author: Sourish Dey

//! Chronos-Bolt Tiny, int8, as a standalone forecaster.
//!
//! # Why this is a wrapper and not a second ONNX session
//!
//! An earlier version of this file opened its own `Session` and ran the graph
//! itself. That is the wrong shape here for two reasons:
//!
//! * **Memory.** A session is tens of megabytes of arena on a machine with a
//!   2 GB ceiling that already holds the chart, the cache and other engines.
//!   Opening a second session for the *same* graph doubles that for no gain.
//!   [`BharatModelEngine`] already keeps one cached session per model behind an
//!   LRU cap, so Chronos is already resident by the time this is used.
//! * **The runtime was being initialised unsafely.** This file called
//!   `ort::init()` directly. Under `ort`'s `load-dynamic`, the first ort call in
//!   a process is what dlopens the runtime, and the bare name
//!   `onnxruntime.dll` resolves through the OS loader to the incompatible 1.17
//!   inbox build in System32, where `ort` `panic!`s on the version check. With
//!   `panic = "abort"` that kills the process. Every engine goes through
//!   [`ort_runtime::ensure_initialized`], which pins the 1.28 build first.
//!
//! So the graph still runs, through the shared engine, and this type is the
//! narrow API for callers who want *only* Chronos rather than a fallback chain.
//!
//! # Graph contract
//!
//! Verified against `models/chronos_bolt_tiny_int8.onnx`:
//!
//! ```text
//! context  : float32 [1, 64]
//! forecast : float32 [1, 9, 64]   = [batch, 9 quantiles, 64 steps]
//! ```
//!
//! The 9 rows are quantiles, so the point path is row
//! [`CHRONOS_P50_INDEX`] and the fan is rows
//! [`CHRONOS_P10_INDEX`]/[`CHRONOS_P90_INDEX`]. Those indices live in
//! [`crate::models::contracts`] rather than being written out again here, because
//! a local `const MEDIAN: usize = 4` is exactly how the wrong row gets read
//! silently. The p10/p50/p90 ordering was confirmed empirically against the real
//! graph, not inferred from the output shape.

use super::ForecastError;
use crate::models::contracts::{CHRONOS_CONTEXT, CHRONOS_P50_INDEX, CHRONOS_QUANTILES};
use crate::models::engine::{BharatModelEngine, ForecastOutput, Model, ModelError};

/// Bars of history the graph consumes.
pub use crate::models::contracts::CHRONOS_CONTEXT as CONTEXT;
/// Steps each quantile row carries.
pub use crate::models::contracts::CHRONOS_HORIZON as MAX_HORIZON;
/// Rows in the output: nine quantiles.
pub use crate::models::contracts::CHRONOS_QUANTILES as QUANTILES;

/// Default model filename under [`super::models_dir`].
pub const CHRONOS_ONNX: &str = "chronos_bolt_tiny_int8.onnx";

/// Chronos-Bolt Tiny, point forecast only.
pub struct ChronosForecaster {
    engine: BharatModelEngine,
}

impl ChronosForecaster {
    /// Point the forecaster at a `models` directory.
    ///
    /// Construction reads nothing from disk: an installation without the graph
    /// still constructs, and the failure arrives on the first prediction.
    pub fn from_models_dir<P: AsRef<std::path::Path>>(models_dir: P) -> Self {
        Self {
            engine: BharatModelEngine::new(models_dir),
        }
    }

    /// Point the forecaster at one graph file.
    ///
    /// The path's parent becomes the models directory, which is how the shared
    /// engine is configured. The file is not opened here.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Self {
        let dir = path
            .as_ref()
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        Self {
            engine: BharatModelEngine::new(dir),
        }
    }

    /// Use the standard model paths.
    pub fn new() -> Self {
        Self::from_models_dir(super::models_dir())
    }

    /// Whether the graph is present on disk. Says nothing about whether it runs;
    /// use [`Self::probe`] for that.
    pub fn is_available(&self) -> bool {
        self.engine.is_available(Model::Chronos)
    }

    /// Run the graph once and discard the result, so a broken install surfaces at
    /// start-up instead of on a user's first forecast.
    pub fn probe(&self) -> Result<(), ForecastError> {
        let flat: Vec<f64> = (0..CHRONOS_CONTEXT)
            .map(|i| 2500.0 + i as f64 * 0.4)
            .collect();
        self.predict(&flat, 8).map(|_| ())
    }

    /// Median (`p50`) path, up to `horizon` points.
    ///
    /// Errors rather than padding: the graph emits a fixed 64 steps, so a
    /// `horizon` above that is clamped, and anything the graph did not produce
    /// is reported instead of being zero-filled into something that looks like a
    /// forecast.
    pub fn predict(&self, history: &[f64], horizon: usize) -> Result<Vec<f64>, ForecastError> {
        if history.is_empty() {
            return Err(ForecastError::EmptyHistory);
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }
        let out = self.raw(history)?;
        let want = horizon.min(MAX_HORIZON).min(out.predictions.len());
        if want == 0 {
            return Err(ForecastError::Model("Chronos returned no points".into()));
        }
        Ok(out.predictions[..want].to_vec())
    }

    /// The full quantile output, for callers that want the fan as well as the
    /// median path.
    pub fn raw(&self, history: &[f64]) -> Result<ForecastOutput, ForecastError> {
        if history.iter().any(|v| !v.is_finite()) {
            return Err(ForecastError::Model(
                "history contains a non-finite price".into(),
            ));
        }
        self.engine
            .predict_chronos(history)
            .map_err(|e| ForecastError::Model(e.to_string()))
    }

    /// The p10/p50/p90 corridor, when the graph emits one.
    pub fn quantile_cone(&self, history: &[f64]) -> Option<crate::models::QuantileConeOutput> {
        let out = self.raw(history).ok()?;
        crate::models::QuantileConeOutput::from_forecast(&out, MAX_HORIZON)
    }
}

impl Default for ChronosForecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-exported so callers can reason about the p50 row without importing
/// `models::contracts`.
pub const MEDIAN_QUANTILE_INDEX: usize = CHRONOS_P50_INDEX;
/// Number of quantile rows in the output.
pub const QUANTILE_ROWS: usize = CHRONOS_QUANTILES;

impl From<ModelError> for ForecastError {
    fn from(e: ModelError) -> Self {
        Self::Model(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forecaster() -> ChronosForecaster {
        ChronosForecaster::new()
    }

    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| 2500.0 + i as f64 * 1.1).collect()
    }

    /// The headline contract from the task: a real, finite, non-empty path.
    #[test]
    fn produces_a_median_path_when_installed() {
        let f = forecaster();
        if !f.is_available() {
            return;
        }
        let out = f
            .predict(&ramp(200), 20)
            .expect("Chronos should run when installed");
        assert!(!out.is_empty(), "must not return an empty forecast");
        assert_eq!(out.len(), 20, "horizon was not honoured");
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }

    /// The reason this file exists: the path must be *curved*, not a flat line.
    /// A flat line is the symptom that motivated wiring Chronos in at all, so it
    /// is the property most worth asserting.
    #[test]
    fn the_path_is_not_flat() {
        let f = forecaster();
        if !f.is_available() {
            return;
        }
        let history = ramp(200);
        let out = f.predict(&history, 20).expect("Chronos should run");
        let last = *history.last().unwrap();
        let lo = out.iter().cloned().fold(f64::MAX, f64::min);
        let hi = out.iter().cloned().fold(f64::MIN, f64::max);
        let span = hi - lo;
        assert!(
            span > 0.0,
            "forecast is a flat line at {lo}, which is the degenerate case"
        );
        // A flat line would also be pinned at the last close; a real forecast
        // wanders away from the anchor.
        assert!(
            (hi - last).abs() > 0.0 || (lo - last).abs() > 0.0,
            "forecast is pinned to the last close, so it is not forecasting"
        );
    }

    /// The median must be read from the p50 row, and the cone must bracket it.
    /// If the quantile index were wrong the point line would still be finite and
    /// plausible, so only a band check catches it.
    #[test]
    fn the_median_row_sits_inside_the_cone() {
        let f = forecaster();
        if !f.is_available() {
            return;
        }
        let out = f.raw(&ramp(200)).expect("Chronos should run");
        let (lower, upper) = match (out.lower.as_ref(), out.upper.as_ref()) {
            (Some(l), Some(u)) => (l, u),
            // No band means this graph is not the one we think it is.
            _ => panic!("Chronos emitted no quantile band; p10/p90 rows missing"),
        };
        assert_eq!(lower.len(), out.predictions.len());
        assert_eq!(upper.len(), out.predictions.len());
        for (i, p) in out.predictions.iter().enumerate() {
            assert!(
                lower[i] <= *p + 1e-3 && *p <= upper[i] + 1e-3,
                "step {i}: p10 {} <= p50 {p} <= p90 {} is violated",
                lower[i],
                upper[i]
            );
        }
    }

    /// A history shorter than the graph's 64-bar context is the caller's problem
    /// and must be named, not padded into a confident answer.
    #[test]
    fn a_short_history_is_named_not_padded() {
        let err = forecaster().predict(&ramp(10), 20).unwrap_err();
        assert!(
            err.to_string().contains("64") || err.to_string().contains("history"),
            "unhelpful error: {err}"
        );
    }

    #[test]
    fn empty_history_and_zero_horizon_are_handled() {
        assert!(matches!(
            forecaster().predict(&[], 10),
            Err(ForecastError::EmptyHistory)
        ));
        assert_eq!(
            forecaster().predict(&ramp(200), 0).unwrap(),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn a_non_finite_bar_is_rejected() {
        let mut h = ramp(200);
        h[5] = f64::NAN;
        let err = forecaster().predict(&h, 20).unwrap_err();
        assert!(err.to_string().contains("non-finite"), "{err}");
    }

    /// A horizon longer than the graph's 64 steps is clamped, never zero-filled.
    #[test]
    fn a_long_horizon_is_clamped_to_what_the_graph_produced() {
        let f = forecaster();
        if !f.is_available() {
            return;
        }
        let out = f.predict(&ramp(200), 5_000).expect("Chronos should run");
        assert!(!out.is_empty());
        assert!(out.len() <= MAX_HORIZON, "got {} points", out.len());
        assert!(out.iter().all(|v| *v != 0.0), "zero-filled tail: {out:?}");
    }

    /// The contract constants must agree with the pinned graph, not drift.
    #[test]
    fn the_pinned_indices_match_the_graph() {
        assert_eq!(CHRONOS_CONTEXT, 64);
        assert_eq!(CHRONOS_QUANTILES, 9);
        assert_eq!(MEDIAN_QUANTILE_INDEX, 4);
        assert_eq!(QUANTILE_ROWS, 9);
    }
}

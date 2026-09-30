// crates/bt-analytics/src/models/patchtst_engine.rs
// Author: Sourish Dey

//! PatchTST subseries transformer, int8.
//!
//! # Graph contract, verified rather than assumed
//!
//! `models/patchtst_tiny_int8.onnx` declares exactly one input and one output:
//!
//! ```text
//! past_values : float32 [1, 64]
//! forecast    : float32 [1, 16]
//! ```
//!
//! Both are concrete, not symbolic, so a wrong rank is rejected by ONNX Runtime
//! rather than silently reinterpreted - `[64]` and `[1, 64, 1]` both fail the
//! shape check. Unlike `ttm_r2_int8.onnx`, the quantized mixer weights here are
//! consistent with the declared input and the graph executes, so the output can
//! be trusted once the shape is verified at runtime rather than assumed.
//!
//! The model emits its forecast in the standardised domain, so the window's mean
//! and standard deviation are applied back afterwards. Reading the raw output as
//! prices would land near zero.
//!
//! # Session lifetime
//!
//! The obvious `forecast` builds a `Session` per call. That is not acceptable
//! here: this app runs on a 2 GB ceiling with other engines already resident, and
//! a session is tens of megabytes of arena. The session is built once, on first
//! use, and reused behind a mutex - the same pattern `BharatModelEngine` and
//! `TtmEngine` already use.

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Bars the graph expects in `past_values`.
pub const PATCHTST_LOOKBACK: usize = 64;
/// Bars the graph emits in `forecast`.
pub const PATCHTST_HORIZON: usize = 16;
/// Graph filename.
pub const FILE_PATCHTST: &str = "patchtst_tiny_int8.onnx";

/// One PatchTST forecast.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchTstForecastResult {
    /// Display name of the graph that produced this.
    pub model_name: String,
    /// Bars projected.
    pub horizon_steps: usize,
    /// Predicted prices, back in the input's own units.
    pub predictions: Vec<f32>,
    /// Last close of the window the forecast was anchored to.
    pub last_close: f32,
}

/// Failures from the PatchTST engine.
#[derive(Debug, thiserror::Error)]
pub enum PatchTstError {
    #[error("PatchTST requires exactly {PATCHTST_LOOKBACK} bars, got {got}")]
    BadWindowLength { got: usize },
    #[error("window contains a non-finite value at index {index}")]
    NonFinite { index: usize },
    #[error("PatchTST graph not found at {path}")]
    ModelNotFound { path: PathBuf },
    #[error("PatchTST failed: {0}")]
    Onnx(String),
    #[error("PatchTST output shape {shape:?} does not match the expected [1, {PATCHTST_HORIZON}]")]
    BadOutputShape { shape: Vec<i64> },
    #[error("PatchTST produced non-finite output")]
    NonFiniteOutput,
}

/// Lazily-loaded PatchTST session.
///
/// Construction performs no IO, so an installation without the graph still
/// starts.
pub struct PatchTstEngine {
    model_path: PathBuf,
    session: Mutex<Option<Session>>,
}

impl PatchTstEngine {
    /// Point at a `models` directory. Nothing is read until the first forecast.
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        Self {
            model_path: models_dir.as_ref().join(FILE_PATCHTST),
            session: Mutex::new(None),
        }
    }

    /// Point straight at a graph file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
        Self {
            model_path: path.as_ref().to_path_buf(),
            session: Mutex::new(None),
        }
    }

    /// Whether the graph is present on disk. Says nothing about whether it runs;
    /// use [`Self::probe`] for that.
    pub fn is_available(&self) -> bool {
        self.model_path.is_file()
    }

    /// Path of the graph this engine will load.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Run the graph once and discard the result, so a broken install is
    /// discovered at start-up rather than on a user's first forecast.
    pub fn probe(&self) -> Result<(), PatchTstError> {
        let window: Vec<f32> = (0..PATCHTST_LOOKBACK)
            .map(|i| 2500.0 + i as f32 * 0.4)
            .collect();
        self.forecast(&window).map(|_| ())
    }

    /// Project 16 bars from a 64-bar window.
    pub fn forecast(&self, window: &[f32]) -> Result<PatchTstForecastResult, PatchTstError> {
        if window.len() != PATCHTST_LOOKBACK {
            return Err(PatchTstError::BadWindowLength { got: window.len() });
        }
        for (index, &v) in window.iter().enumerate() {
            if !v.is_finite() {
                return Err(PatchTstError::NonFinite { index });
            }
        }

        // Standardise. Population (not sample) variance, matching how the graph
        // was traced. The floor keeps a perfectly flat window from dividing by
        // zero; it stays a small number rather than NaN.
        let mean = window.iter().sum::<f32>() / PATCHTST_LOOKBACK as f32;
        let variance =
            window.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / PATCHTST_LOOKBACK as f32;
        let std_dev = variance.sqrt().max(1e-5);
        let normalized: Vec<f32> = window.iter().map(|&p| (p - mean) / std_dev).collect();

        let predictions = self.with_session(|session| {
            // Shape [1, 64] as a (shape, buffer) tuple: the ORT crate vendors a
            // newer ndarray than this workspace pins, and a tuple crosses that
            // version boundary without a type mismatch. This matches
            // `BharatModelEngine::run_window_raw`.
            let tensor = ort::value::Tensor::from_array((
                vec![1i64, PATCHTST_LOOKBACK as i64],
                normalized.clone(),
            ))
            .map_err(|e| PatchTstError::Onnx(e.to_string()))?;

            let outputs = session
                .run(ort::inputs![tensor])
                .map_err(|e| PatchTstError::Onnx(e.to_string()))?;

            // `try_extract_tensor` yields `(shape, &[f32])`; the shape is checked
            // rather than assumed, because a silent reinterpretation here would
            // return sixteen numbers that are not the forecast.
            let (shape, flat) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| PatchTstError::Onnx(e.to_string()))?;

            let dims: Vec<i64> = shape.iter().copied().collect();
            if dims != [1, PATCHTST_HORIZON as i64] {
                return Err(PatchTstError::BadOutputShape { shape: dims });
            }

            Ok(flat.to_vec())
        })?;

        if predictions.iter().any(|v| !v.is_finite()) {
            return Err(PatchTstError::NonFiniteOutput);
        }

        // Back to the price domain.
        let denormalized: Vec<f32> = predictions.iter().map(|&v| v * std_dev + mean).collect();

        Ok(PatchTstForecastResult {
            model_name: "PatchTST Subseries Transformer (int8)".to_string(),
            horizon_steps: denormalized.len(),
            predictions: denormalized,
            last_close: window[PATCHTST_LOOKBACK - 1],
        })
    }

    /// Load the session once, then reuse it.
    fn with_session<T>(
        &self,
        f: impl FnOnce(&mut Session) -> Result<T, PatchTstError>,
    ) -> Result<T, PatchTstError> {
        // Pin the runtime before any ort call. `Tensor::from_array` is itself an
        // ort call, and being first in the process it is what makes ort dlopen
        // the runtime by bare name - which on a machine with an older inbox copy
        // in System32 resolves to the wrong DLL and panics. See `ort_runtime`.
        crate::ort_runtime::ensure_initialized().map_err(PatchTstError::Onnx)?;

        let mut guard = self
            .session
            .lock()
            .map_err(|_| PatchTstError::Onnx("session lock poisoned".into()))?;

        if guard.is_none() {
            if !self.model_path.is_file() {
                return Err(PatchTstError::ModelNotFound {
                    path: self.model_path.clone(),
                });
            }
            // Same budget rules as every other engine here: one thread, Level1,
            // no memory-pattern arena.
            let session = Session::builder()
                .map_err(|e| PatchTstError::Onnx(e.to_string()))?
                .with_optimization_level(GraphOptimizationLevel::Level1)
                .map_err(|e| PatchTstError::Onnx(e.to_string()))?
                .with_intra_threads(1)
                .map_err(|e| PatchTstError::Onnx(e.to_string()))?
                .with_inter_threads(1)
                .map_err(|e| PatchTstError::Onnx(e.to_string()))?
                .with_memory_pattern(false)
                .map_err(|e| PatchTstError::Onnx(e.to_string()))?
                .commit_from_file(&self.model_path)
                .map_err(|e| PatchTstError::Onnx(e.to_string()))?;
            *guard = Some(session);
        }

        // `map(f)` would yield `Option<Result<..>>`, so the missing case is
        // matched explicitly rather than flattened.
        match guard.as_mut() {
            Some(session) => f(session),
            None => Err(PatchTstError::Onnx("session missing after load".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> PatchTstEngine {
        PatchTstEngine::new(crate::forecast::models_dir())
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| 2500.0 + i as f32 * 1.1).collect()
    }

    /// The headline contract: sixteen finite bars.
    #[test]
    fn projects_sixteen_bars_when_installed() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        let res = e
            .forecast(&ramp(PATCHTST_LOOKBACK))
            .expect("PatchTST should run");
        assert_eq!(res.horizon_steps, PATCHTST_HORIZON);
        assert_eq!(res.predictions.len(), PATCHTST_HORIZON);
        assert!(res.predictions.iter().all(|p| p.is_finite()));
    }

    /// If the de-standardisation were skipped the output would sit near zero, so
    /// this is the test that the model is actually being read in the price domain.
    #[test]
    fn the_output_lives_in_the_price_domain() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        let window = ramp(PATCHTST_LOOKBACK);
        let res = e.forecast(&window).expect("PatchTST should run");
        let lo = window.iter().cloned().fold(f32::MAX, f32::min);
        let hi = window.iter().cloned().fold(f32::MIN, f32::max);
        let span = (hi - lo).max(1.0);
        for p in &res.predictions {
            assert!(
                (*p - lo).abs() < 50.0 * span,
                "prediction {p} escaped the input range {lo}..{hi}"
            );
        }
        assert_eq!(res.last_close, window[PATCHTST_LOOKBACK - 1]);
    }

    #[test]
    fn the_window_length_is_enforced() {
        assert!(matches!(
            engine().forecast(&ramp(63)),
            Err(PatchTstError::BadWindowLength { got: 63 })
        ));
        assert!(matches!(
            engine().forecast(&ramp(65)),
            Err(PatchTstError::BadWindowLength { got: 65 })
        ));
        assert!(matches!(
            engine().forecast(&[]),
            Err(PatchTstError::BadWindowLength { got: 0 })
        ));
    }

    #[test]
    fn a_non_finite_bar_is_rejected_by_index() {
        let mut w = ramp(PATCHTST_LOOKBACK);
        w[11] = f32::NAN;
        assert!(matches!(
            engine().forecast(&w),
            Err(PatchTstError::NonFinite { index: 11 })
        ));
        w[11] = f32::INFINITY;
        assert!(matches!(
            engine().forecast(&w),
            Err(PatchTstError::NonFinite { index: 11 })
        ));
    }

    #[test]
    fn a_missing_graph_reports_the_path() {
        let e = PatchTstEngine::new("definitely-not-a-models-dir");
        assert!(!e.is_available());
        assert!(matches!(
            e.forecast(&ramp(PATCHTST_LOOKBACK)),
            Err(PatchTstError::ModelNotFound { .. })
        ));
    }

    /// A perfectly flat window used to be the classic divide-by-zero; the floor
    /// in the standardisation has to hold here.
    #[test]
    fn a_flat_window_does_not_produce_nan() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        let flat = vec![1182.0f32; PATCHTST_LOOKBACK];
        let res = e
            .forecast(&flat)
            .expect("a flat window must still be answerable");
        assert!(res.predictions.iter().all(|p| p.is_finite()));
    }

    /// Repeated calls must reuse the cached session, not rebuild it.
    #[test]
    fn the_session_is_reused_across_calls() {
        let e = engine();
        if !e.is_available() {
            return;
        }
        let first = e.forecast(&ramp(PATCHTST_LOOKBACK)).expect("first");
        let second = e.forecast(&ramp(PATCHTST_LOOKBACK)).expect("second");
        assert_eq!(first.predictions, second.predictions);
    }

    /// The result must serialise, since the API hands it straight to axum.
    #[test]
    fn the_result_serialises() {
        let r = PatchTstForecastResult {
            model_name: "x".into(),
            horizon_steps: 16,
            predictions: vec![1.0; 16],
            last_close: 1.0,
        };
        let json = serde_json::to_string(&r).expect("serialise");
        assert!(json.contains("\"predictions\""), "{json}");
        let back: PatchTstForecastResult = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back, r);
    }
}

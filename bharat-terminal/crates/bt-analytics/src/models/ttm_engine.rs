// crates/bt-analytics/src/models/ttm_engine.rs
// Author: Sourish Dey

//! IBM TinyTimeMixer R2 (int8): 512 bars in, 96 bars out.
//!
//! # Verified contract
//!
//! Read off the graph rather than assumed:
//!
//! ```text
//! input  past_values : tensor(float) ['batch_size', 512, 1]
//! output predictions : tensor(float) ['batch_size', 96, 'Addpredictions_dim_2']
//! ```
//!
//! So the input is `[1, 512, 1]` - a single univariate channel - and the output is
//! 96 steps. Anything that passes `[1, 512]` fails the shape check, and anything
//! that treats the output as a plain `[1, 96]` mis-reads the trailing channel.
//!
//! # Preprocessing
//!
//! The window is standardised to zero mean and unit variance, and the prediction
//! is mapped back with the same statistics. A flat window would divide by zero,
//! so the scale is floored; that yields the mean as a constant path, which is the
//! correct answer for input with no variation.
//!
//! # Memory
//!
//! The session is built with `intra_threads = inter_threads = 1`,
//! `GraphOptimizationLevel::Level1` and the arena memory pattern disabled, and it
//! is cached so the 1.2 MB graph is not re-read per call. On a 2 GB machine that
//! is the difference between a bounded working set and a slow leak.

use super::engine::{ensure_finite, ModelError};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::{Path, PathBuf};

/// Bars TinyTimeMixer requires.
pub const TTM_CONTEXT: usize = 512;

/// Steps TinyTimeMixer emits.
pub const TTM_HORIZON: usize = 96;

/// File name of the exported graph.
pub const FILE_TTM_R2: &str = "ttm_r2_int8.onnx";

/// One TinyTimeMixer run.
#[derive(Debug, Clone, PartialEq)]
pub struct TtmForecastResult {
    /// Display name of the model that produced this.
    pub model_name: String,
    /// Bars projected.
    pub horizon_steps: usize,
    /// Projected closes, in the input's own price units.
    pub predictions: Vec<f64>,
}

impl TtmForecastResult {
    /// Final projected price, which is what a narrative usually quotes.
    pub fn terminal(&self) -> Option<f64> {
        self.predictions.last().copied()
    }

    /// Projected change from `anchor` to the last projection, as a fraction.
    /// `None` when the anchor is not a usable price.
    pub fn projected_change(&self, anchor: f64) -> Option<f64> {
        let last = self.terminal()?;
        if anchor.abs() < 1e-12 {
            return None;
        }
        Some(last / anchor - 1.0)
    }
}

/// Standardise a window and return it with the statistics needed to invert it.
fn standardize(window: &[f64]) -> (Vec<f32>, f64, f64) {
    let n = window.len().max(1) as f64;
    let mean = window.iter().sum::<f64>() / n;
    let variance = window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    // A perfectly flat window has zero scale. Flooring it keeps the input
    // finite; the normalised window is then all zeros and the model returns the
    // mean, which is the honest forecast for constant input.
    let sd = if variance.is_finite() && variance > 1e-24 {
        variance.sqrt()
    } else {
        1.0
    };
    let normalized = window.iter().map(|v| ((v - mean) / sd) as f32).collect();
    (normalized, mean, sd)
}

/// Lazily-loaded TinyTimeMixer session.
///
/// Construction performs no IO, so a machine without the file still starts.
pub struct TtmEngine {
    model_path: PathBuf,
    session: std::sync::Mutex<Option<Session>>,
    /// One-shot verdict on whether this graph can execute at all.
    capability: std::sync::Mutex<Option<Result<(), String>>>,
}

/// Verdict from [`TtmEngine::probe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtmStatus {
    /// The graph is present and runs.
    Ready,
    /// The file is not installed.
    Missing,
    /// The file is installed but the graph itself fails to execute.
    ///
    /// This is not a shape problem on our side. `ttm_r2_int8.onnx` as shipped
    /// fails on every input at every optimisation level with `MatMulInteger`
    /// "dimension mismatch": its quantized mixer weights are statically shaped
    /// ([36, 72] for patch_mixer, [48, 96] for feature_mixer) to a context the
    /// declared `[1, 512, 1]` input cannot produce. The export is inconsistent
    /// with itself, so the only correct behaviour is to say so rather than emit
    /// numbers.
    GraphBroken(String),
}

impl TtmEngine {
    /// Point at a `models` directory. Nothing is read until the first forecast.
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        Self {
            model_path: models_dir.as_ref().join(FILE_TTM_R2),
            session: std::sync::Mutex::new(None),
            capability: std::sync::Mutex::new(None),
        }
    }

    /// Whether the graph is present on disk. Says nothing about whether it runs;
    /// use [`Self::probe`] for that.
    pub fn is_available(&self) -> bool {
        self.model_path.exists()
    }

    /// Path of the graph this engine will load.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Run the graph once on a constant window and cache the verdict.
    ///
    /// Callers use this to decide whether to offer TinyTimeMixer at all, rather
    /// than discovering on every forecast that the graph is unusable. The probe
    /// uses a flat window, which is the cheapest input that still exercises every
    /// node.
    pub fn probe(&self) -> TtmStatus {
        if !self.is_available() {
            return TtmStatus::Missing;
        }
        if let Some(hit) = self.capability.lock().ok().and_then(|g| g.clone()) {
            return match hit {
                Ok(()) => TtmStatus::Ready,
                Err(m) => TtmStatus::GraphBroken(m),
            };
        }
        let verdict = self
            .forecast_flat_probe()
            .map_err(|e: ModelError| e.to_string());
        if let Ok(mut g) = self.capability.lock() {
            *g = Some(verdict.clone());
        }
        match verdict {
            Ok(()) => TtmStatus::Ready,
            Err(m) => TtmStatus::GraphBroken(m),
        }
    }

    fn forecast_flat_probe(&self) -> Result<(), ModelError> {
        self.forecast(&vec![1.0; TTM_CONTEXT]).map(|_| ())
    }

    fn with_session<T>(
        &self,
        f: impl FnOnce(&mut Session) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        let fail = |m: String| ModelError::Onnx {
            model: "TinyTimeMixer R2 (int8)",
            message: m,
        };
        let mut guard = self
            .session
            .lock()
            .map_err(|_| fail("session lock poisoned".into()))?;
        if guard.is_none() {
            if !self.is_available() {
                return Err(ModelError::ModelNotFound(FILE_TTM_R2.to_string()));
            }
            crate::ort_runtime::ensure_initialized().map_err(ModelError::Runtime)?;
            let s = Session::builder()
                .map_err(|e| fail(e.to_string()))?
                .with_optimization_level(GraphOptimizationLevel::Level1)
                .map_err(|e| fail(e.to_string()))?
                .with_intra_threads(1)
                .map_err(|e| fail(e.to_string()))?
                .with_inter_threads(1)
                .map_err(|e| fail(e.to_string()))?
                .with_memory_pattern(false)
                .map_err(|e| fail(e.to_string()))?
                .commit_from_file(&self.model_path)
                .map_err(|e| fail(e.to_string()))?;
            *guard = Some(s);
        }
        guard
            .as_mut()
            .map(f)
            .unwrap_or_else(|| Err(fail("session missing after load".into())))
    }

    /// Project 96 bars from a 512-bar window.
    pub fn forecast(&self, window: &[f64]) -> Result<TtmForecastResult, ModelError> {
        if window.len() < TTM_CONTEXT {
            return Err(ModelError::InsufficientHistory {
                needed: TTM_CONTEXT,
                got: window.len(),
            });
        }
        if window.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::Onnx {
                model: "TinyTimeMixer R2 (int8)",
                message: "window contains non-finite prices".into(),
            });
        }
        let window = &window[window.len() - TTM_CONTEXT..];
        let (normalized, mean, sd) = standardize(window);

        // Shape [1, 512, 1]: one series, one channel. Passed as a
        // (shape, flat-buffer) tuple rather than an ndarray because the ort crate
        // vendors a newer ndarray than this workspace pins.
        let flat: Vec<f32> = normalized.iter().flat_map(|v| [*v]).collect();
        let tensor = ort::value::Tensor::from_array((vec![1i64, TTM_CONTEXT as i64, 1i64], flat))
            .map_err(|e| ModelError::Onnx {
            model: "TinyTimeMixer R2 (int8)",
            message: e.to_string(),
        })?;

        // Inference and extraction both happen inside the closure: `run` returns values
        // that borrow the session, so they cannot escape `with_session`.
        let predictions = self.with_session(|session| {
            let outputs = session
                .run(ort::inputs![tensor])
                .map_err(|e| ModelError::Onnx {
                    model: "TinyTimeMixer R2 (int8)",
                    message: e.to_string(),
                })?;
            let (_shape, flat_out) =
                outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| ModelError::Onnx {
                        model: "TinyTimeMixer R2 (int8)",
                        message: e.to_string(),
                    })?;
            // Output is [1, 96, 1]; take the first `TTM_HORIZON` scalars and
            // invert the standardisation.
            Ok(flat_out
                .iter()
                .take(TTM_HORIZON)
                .map(|v| (*v as f64) * sd + mean)
                .collect::<Vec<f64>>())
        })?;
        ensure_finite("TinyTimeMixer R2 (int8)", &predictions)?;

        Ok(TtmForecastResult {
            model_name: "TinyTimeMixer R2 (int8)".to_string(),
            horizon_steps: predictions.len(),
            predictions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_window_is_reported_not_panicked() {
        let e = TtmEngine::new(".");
        assert!(matches!(
            e.forecast(&[100.0; 100]),
            Err(ModelError::InsufficientHistory {
                needed: TTM_CONTEXT,
                got: 100
            })
        ));
    }

    #[test]
    fn a_non_finite_window_is_rejected() {
        let e = TtmEngine::new(".");
        let mut w = vec![100.0; TTM_CONTEXT];
        w[7] = f64::NAN;
        assert!(matches!(e.forecast(&w), Err(ModelError::Onnx { .. })));
    }

    /// A missing model must fail clearly, not panic and not return zeros.
    #[test]
    fn a_missing_model_reports_clearly() {
        let e = TtmEngine::new("definitely-not-a-models-dir");
        assert!(!e.is_available());
        let err = e
            .forecast(&[100.0; TTM_CONTEXT])
            .expect_err("must not succeed");
        assert!(matches!(err, ModelError::ModelNotFound(_)), "got {err:?}");
    }

    #[test]
    fn construction_performs_no_io() {
        let e = TtmEngine::new("definitely-not-a-models-dir");
        assert!(e.model_path().ends_with(FILE_TTM_R2));
        assert!(!e.is_available());
    }

    /// Standardisation must be exactly invertible, or every prediction is biased.
    #[test]
    fn standardisation_round_trips() {
        let window: Vec<f64> = (0..TTM_CONTEXT)
            .map(|i| 100.0 + (i as f64).sin() * 5.0)
            .collect();
        let (norm, mean, sd) = standardize(&window);
        assert_eq!(norm.len(), TTM_CONTEXT);
        let restored: Vec<f64> = norm.iter().map(|v| *v as f64 * sd + mean).collect();
        for (a, b) in window.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn a_flat_window_yields_a_finite_scale() {
        let (norm, mean, sd) = standardize(&vec![2500.0; TTM_CONTEXT]);
        assert!(sd.is_finite() && sd > 0.0);
        assert!(norm.iter().all(|v| v.is_finite()));
        assert!((mean - 2500.0).abs() < 1e-9);
    }

    #[test]
    fn projected_change_ignores_a_nonsense_anchor() {
        let r = TtmForecastResult {
            model_name: "t".into(),
            horizon_steps: 1,
            predictions: vec![100.0],
        };
        assert_eq!(r.projected_change(100.0), Some(0.0));
        assert!(r.projected_change(0.0).is_none());
    }

    /// The probe must reach a definite verdict, never hang or panic, and the
    /// verdict must be cached so the cost is paid once per session.
    #[test]
    fn the_probe_reaches_a_definite_verdict() {
        let e = TtmEngine::new("definitely-not-a-models-dir");
        assert_eq!(e.probe(), TtmStatus::Missing);

        let e = TtmEngine::new(crate::forecast::models_dir());
        let first = e.probe();
        let second = e.probe();
        assert_eq!(first, second, "the probe must be cached");
        match first {
            TtmStatus::Missing => {}
            TtmStatus::Ready => {}
            // A broken graph is a legitimate verdict, provided it is specific
            // enough to act on.
            TtmStatus::GraphBroken(msg) => assert!(
                !msg.is_empty(),
                "a broken-graph verdict must carry the reason"
            ),
        }
    }

    /// Whether or not the graph executes, the engine must fail *cleanly*.
    ///
    /// The shipped `ttm_r2_int8.onnx` does not execute (its quantized mixer
    /// weights are statically shaped to a context its declared `[1, 512, 1]`
    /// input cannot produce), so the honest contract is: an error, a specific
    /// message, and never a silent return of numbers that look like a forecast.
    #[test]
    fn ttm_never_silently_returns_garbage() {
        let e = TtmEngine::new(crate::forecast::models_dir());
        if !e.is_available() {
            return;
        }
        let closes: Vec<f64> = (0..600)
            .map(|i| {
                let t = i as f64;
                2500.0 + t * 0.4 + (t * 0.15).sin() * 6.0
            })
            .collect();
        match e.forecast(&closes) {
            Ok(out) => {
                assert_eq!(out.predictions.len(), TTM_HORIZON);
                assert!(out.predictions.iter().all(|v| v.is_finite()));
                let lo = closes.iter().cloned().fold(f64::MAX, f64::min);
                let hi = closes.iter().cloned().fold(f64::MIN, f64::max);
                let span = (hi - lo).max(1.0);
                for v in &out.predictions {
                    assert!(
                        (*v - lo).abs() < 50.0 * span,
                        "prediction {v} escaped the input range {lo}..{hi}"
                    );
                }
            }
            Err(e) => {
                // The error must name the model and carry a reason, so the UI can
                // say "this engine is unusable" rather than "forecast failed".
                let text = e.to_string();
                assert!(!text.is_empty(), "an error with no message");
                assert!(
                    text.contains("TinyTimeMixer") || text.contains("MatMul"),
                    "unhelpful error: {text}"
                );
            }
        }
    }
}

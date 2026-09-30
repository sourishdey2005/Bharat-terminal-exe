// crates/bt-analytics/src/models/engine.rs
// Author: Sourish Dey

//! Low-memory ONNX inference for the small forecasting models.
//!
//! # Memory discipline (2 GB budget)
//!
//! Every session is built with a single intra-op and a single inter-op thread,
//! graph optimization capped at `Level1`, and `with_memory_pattern(false)`. The
//! last one is the important one: memory-pattern planning builds an internal
//! arena table sized to the graph, and for a machine with 2 GB of RAM those
//! tables are exactly the kind of large pinned block that causes fragmentation
//! and allocation failures later.
//!
//! # Session lifetime
//!
//! Sessions are created **lazily on first use** and then cached, rather than
//! rebuilt per call. Rebuilding would re-read and re-initialize the weights
//! (8.6 MB for the int8 Chronos graph) on every forecast, which is both slow and
//! invites several multi-megabyte sessions to exist at once. The cache holds at
//! most [`MAX_CACHED_SESSIONS`] graphs and evicts least-recently-used, so
//! switching between models over a long session cannot grow without bound.
//!
//! Inference itself is serialized per session behind a mutex. `ort` 2.x takes
//! `&mut self` on `run`, and the app is single-threaded for inference anyway, so
//! a contended lock is cheaper than the alternative and keeps the type `Sync`
//! enough to live in shared app state.
//!
//! # Input contract
//!
//! Windows are normalized per window against the **last close**:
//!
//! ```text
//! x = (window - last) / sd        y = (horizon output is in the same units)
//! forecast_price = last + y * sd
//! ```
//!
//! This is the convention `train_forecasts.py` trains under, so a zero output
//! means "random walk". Both forecasters were selected on a held-out split with
//! the untrained (random-walk) state as a candidate epoch, so neither can
//! regress below the baseline.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use super::sliding_window::SlidingBuffer;

/// Lookback length shared by the two small forecasters.
pub const DLINEAR_LOOKBACK: usize = 32;
/// Forecast steps emitted by DLinear.
pub const DLINEAR_HORIZON: usize = 5;
/// Lookback length shared by N-HiTS and Chronos-Bolt.
pub const NHITS_LOOKBACK: usize = 32;
/// Forecast steps emitted by N-HiTS.
pub const NHITS_HORIZON: usize = 5;
/// Context length Chronos-Bolt was exported with.
pub const CHRONOS_CONTEXT: usize = 64;
/// Forecast steps emitted by Chronos-Bolt.
pub const CHRONOS_HORIZON: usize = 64;
/// Quantile heads in the Chronos-Bolt head.
pub const CHRONOS_QUANTILES: usize = 9;
/// Bars in the signal classifier's OHLCV window.
pub const SIGNAL_WINDOW: usize = 30;

/// Model filenames, resolved inside the models directory.
pub const FILE_DLINEAR: &str = "dlinear.onnx";
pub const FILE_NHITS: &str = "nhits_small.onnx";
pub const FILE_CHRONOS: &str = "chronos_bolt_tiny_int8.onnx";
pub const FILE_CHRONOS_FP32: &str = "chronos_bolt_tiny.onnx";

/// Upper bound on simultaneously cached sessions.
///
/// Chronos int8 is the largest at ~9 MB. Four of these plus the classifier and
/// the 35 MB fp32 fallback would be wasteful on a 2 GB machine, so the cache is
/// capped and evicts least-recently-used.
pub const MAX_CACHED_SESSIONS: usize = 4;

/// Errors surfaced by the engine.
/// Errors surfaced by the engine.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model file not found: {0}")]
    ModelNotFound(String),
    #[error("expected a {expected}-point window, got {got}")]
    WindowLength { expected: usize, got: usize },
    #[error("history too short: need {needed}, got {got}")]
    InsufficientHistory { needed: usize, got: usize },
    #[error("ONNX runtime unavailable: {0}")]
    Runtime(String),
    #[error("{model}: {message}")]
    Onnx {
        model: &'static str,
        message: String,
    },
    #[error("{0} produced non-finite output")]
    NonFinite(&'static str),
}

/// Result of a horizon forecast.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastOutput {
    /// Display name of the model that produced this.
    pub model_name: String,
    /// Number of predicted points.
    pub horizon_steps: usize,
    /// Predicted prices in the input's own units.
    pub predictions: Vec<f64>,
    /// Lower quantile band (10th percentile), when the model emits one.
    pub lower: Option<Vec<f64>>,
    /// Upper quantile band (90th percentile), when the model emits one.
    pub upper: Option<Vec<f64>>,
    /// Input points the model actually consumed.
    pub lookback: usize,
}

/// Buy/Hold/Sell classification.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalOutput {
    /// "BUY", "HOLD" or "SELL".
    pub signal: String,
    /// Probability of the chosen class.
    pub confidence: f32,
    /// Class probabilities in the model's own class order.
    pub probabilities: [f32; 3],
}

/// Which cached model a session belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Slot {
    DLinear,
    NHits,
    Chronos,
}

/// A cached session plus a use counter for LRU eviction.
struct Cached {
    session: Mutex<Session>,
    /// Monotonic tick of last use; higher is more recent.
    last_used: Mutex<u64>,
}

/// Lazily-loaded ONNX engine for the small local models.
///
/// Cheap to construct: nothing is read from disk until a model is first
/// requested, so an installation missing some models still starts normally.
pub struct BharatModelEngine {
    models_dir: PathBuf,
    cache: Mutex<HashMap<Slot, Cached>>,
    clock: Mutex<u64>,
}

impl BharatModelEngine {
    /// Create an engine reading from `models_dir`. Construction performs no IO.
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            cache: Mutex::new(HashMap::new()),
            clock: Mutex::new(0),
        }
    }

    /// Create an engine using the standard `models/` location.
    pub fn with_default_paths() -> Self {
        Self::new(crate::forecast::models_dir())
    }

    /// The directory this engine reads from.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Whether a model's file is present, without loading it.
    pub fn is_available(&self, model: Model) -> bool {
        self.resolve(model).is_some()
    }

    /// Models whose files are present, in display order.
    pub fn available_models(&self) -> Vec<Model> {
        Model::ALL
            .iter()
            .copied()
            .filter(|m| self.is_available(*m))
            .collect()
    }

    /// Resolve a model to a concrete file, preferring the int8 Chronos build.
    fn resolve(&self, model: Model) -> Option<PathBuf> {
        let candidates: &[&str] = match model {
            Model::DLinear => &[FILE_DLINEAR],
            Model::NHiTS => &[FILE_NHITS],
            // The int8 head is the shipped default; the fp32 graph is a
            // fallback for anyone who wants full precision at 35 MB.
            Model::Chronos => &[FILE_CHRONOS, FILE_CHRONOS_FP32],
        };
        candidates
            .iter()
            .map(|f| self.models_dir.join(f))
            .find(|p| p.is_file())
    }

    /// Build a session tuned for the 2 GB ceiling.
    fn build_session(model: Model, path: &Path) -> Result<Session, ModelError> {
        // Pinned runtime only: a blind init could dlopen an incompatible system
        // build and crash natively instead of returning an error.
        crate::ort_runtime::ensure_initialized().map_err(ModelError::Runtime)?;

        let label = model.label();
        let fail = |message: String| ModelError::Onnx {
            model: label,
            message,
        };

        // CRITICAL for 2 GB RAM: single threads, Level1 only, and no arena
        // memory pattern, otherwise the allocator can pin large blocks.
        let session = Session::builder()
            .map_err(|e| fail(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| fail(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| fail(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| fail(e.to_string()))?
            .with_memory_pattern(false)
            .map_err(|e| fail(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| fail(e.to_string()))?;

        tracing::info!("{} ready: {}", label, path.display());
        Ok(session)
    }

    /// Return a session for `model`, loading it on first use.
    ///
    /// Evicts the least-recently-used entry when the cache is full, so the
    /// resident set stays bounded no matter how many models get cycled.
    fn session(&self, model: Model) -> Result<(), ModelError> {
        let slot = model.slot();
        if self.cache.lock().map_err(poisoned)?.contains_key(&slot) {
            self.touch(slot)?;
            return Ok(());
        }

        let path = self
            .resolve(model)
            .ok_or_else(|| ModelError::ModelNotFound(model.label().to_string()))?;
        let session = Self::build_session(model, &path)?;

        let mut cache = self.cache.lock().map_err(poisoned)?;
        if cache.len() >= MAX_CACHED_SESSIONS {
            let victim = cache
                .iter()
                .min_by_key(|(_, c)| c.last_used.lock().map(|g| *g).unwrap_or(0))
                .map(|(k, _)| *k);
            if let Some(v) = victim {
                tracing::debug!("evicting {:?} from the ONNX session cache", v);
                cache.remove(&v);
            }
        }
        cache.insert(
            slot,
            Cached {
                session: Mutex::new(session),
                last_used: Mutex::new(0),
            },
        );
        drop(cache);
        self.touch(slot)?;
        Ok(())
    }

    /// Stamp a slot as most recently used.
    fn touch(&self, slot: Slot) -> Result<(), ModelError> {
        let tick = {
            let mut clock = self.clock.lock().map_err(poisoned)?;
            *clock += 1;
            *clock
        };
        let cache = self.cache.lock().map_err(poisoned)?;
        if let Some(entry) = cache.get(&slot) {
            if let Ok(mut last) = entry.last_used.lock() {
                *last = tick;
            }
        }
        Ok(())
    }

    /// Run one `(1, n)` window through a model and return the raw output.
    fn run_window(&self, model: Model, window: &[f32]) -> Result<Vec<f32>, ModelError> {
        self.run_window_raw(model, window)
    }

    /// Shared inference path: load (or reuse) the session, run the window, and
    /// return the flattened first output.
    fn run_window_raw(&self, model: Model, window: &[f32]) -> Result<Vec<f32>, ModelError> {
        self.session(model)?;
        let label = model.label();

        // Passed as a (shape, flat-buffer) tuple rather than an ndarray: the ORT
        // crate vendors a newer ndarray than this workspace, and a tuple crosses
        // that version boundary without a type mismatch.
        let tensor =
            ort::value::Tensor::from_array((vec![1i64, window.len() as i64], window.to_vec()))
                .map_err(|e| ModelError::Onnx {
                    model: label,
                    message: e.to_string(),
                })?;

        let mut cache = self.cache.lock().map_err(poisoned)?;
        let entry = cache
            .get_mut(&model.slot())
            .ok_or_else(|| ModelError::Onnx {
                model: label,
                message: "session evicted".into(),
            })?;
        let mut session = entry.session.lock().map_err(|_| ModelError::Onnx {
            model: label,
            message: "session lock poisoned".into(),
        })?;

        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| ModelError::Onnx {
                model: label,
                message: e.to_string(),
            })?;
        let (_, first) = outputs.iter().next().ok_or_else(|| ModelError::Onnx {
            model: label,
            message: "no outputs".into(),
        })?;
        let view = first
            .try_extract_array::<f32>()
            .map_err(|e| ModelError::Onnx {
                model: label,
                message: e.to_string(),
            })?;
        Ok(view.iter().copied().collect())
    }

    /// DLinear: 32 lookback points to a 5 step forecast.
    pub fn predict_dlinear(&self, closes: &[f64]) -> Result<ForecastOutput, ModelError> {
        self.predict_small(Model::DLinear, closes, DLINEAR_HORIZON)
    }

    /// N-HiTS (small): 32 lookback points to a 5 step forecast.
    pub fn predict_nhits(&self, closes: &[f64]) -> Result<ForecastOutput, ModelError> {
        self.predict_small(Model::NHiTS, closes, NHITS_HORIZON)
    }

    /// Run any of the three window-based models by enum value.
    ///
    /// This is the entry point the forecast chain uses, so a single dispatch
    /// covers DLinear, N-HiTS and Chronos instead of three near-identical
    /// match arms at every call site.
    pub fn run(&self, model: Model, closes: &[f64]) -> Result<ForecastOutput, ModelError> {
        match model {
            Model::DLinear => self.predict_dlinear(closes),
            Model::NHiTS => self.predict_nhits(closes),
            Model::Chronos => self.predict_chronos(closes),
        }
    }

    /// Shared path for the two 32-point models.
    fn predict_small(
        &self,
        model: Model,
        closes: &[f64],
        horizon: usize,
    ) -> Result<ForecastOutput, ModelError> {
        let (window, anchor, sd) = normalize_window(closes, DLINEAR_LOOKBACK)?;
        let raw = self.run_window(model, &window)?;

        let predictions: Vec<f64> = raw
            .iter()
            .take(horizon)
            .map(|v| anchor + *v as f64 * sd)
            .collect();
        ensure_finite(model.label(), &predictions)?;

        Ok(ForecastOutput {
            model_name: model.label().to_string(),
            horizon_steps: predictions.len(),
            predictions,
            lower: None,
            upper: None,
            lookback: DLINEAR_LOOKBACK,
        })
    }

    /// Chronos-Bolt Tiny (int8): 64 context points to a 64 step quantile
    /// forecast.
    ///
    /// Unlike the two small models, Chronos is fed **raw prices** and returns
    /// **raw prices**. It carries its own `InstanceNorm` inside the graph, so
    /// normalizing the input here would be a second normalization and the output
    /// would come back in the wrong units. This asymmetry is deliberate and is
    /// the single easiest thing to get wrong when calling these three models
    /// side by side.
    ///
    /// The head emits `[1, 9, 64]`: 9 quantile levels. The median (index 4) is
    /// the point forecast and the 10th/90th levels form the band, which is the
    /// reason to prefer a quantile model over a point one.
    pub fn predict_chronos(&self, closes: &[f64]) -> Result<ForecastOutput, ModelError> {
        if closes.len() < CHRONOS_CONTEXT {
            return Err(ModelError::InsufficientHistory {
                needed: CHRONOS_CONTEXT,
                got: closes.len(),
            });
        }
        let tail = &closes[closes.len() - CHRONOS_CONTEXT..];
        let raw_closes: Vec<f32> = tail.iter().map(|v| *v as f32).collect();
        if raw_closes.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::Onnx {
                model: Model::Chronos.label(),
                message: "context contains non-finite prices".into(),
            });
        }

        let raw = self.run_window_raw(Model::Chronos, &raw_closes)?;

        // Flattened [quantiles, horizon]; guard a flat head defensively.
        if raw.len() < CHRONOS_QUANTILES * CHRONOS_HORIZON {
            return Err(ModelError::Onnx {
                model: Model::Chronos.label(),
                message: format!(
                    "expected {} values, got {}",
                    CHRONOS_QUANTILES * CHRONOS_HORIZON,
                    raw.len()
                ),
            });
        }

        let band = |q: usize| -> Vec<f64> {
            let start = q * CHRONOS_HORIZON;
            raw[start..start + CHRONOS_HORIZON]
                .iter()
                .map(|v| *v as f64)
                .collect()
        };

        // Index 0 is the 10th percentile, index 8 the 90th.
        let predictions = band(CHRONOS_QUANTILES / 2);
        let lower = band(0);
        let upper = band(CHRONOS_QUANTILES - 1);
        ensure_finite(Model::Chronos.label(), &predictions)?;
        ensure_finite(Model::Chronos.label(), &lower)?;
        ensure_finite(Model::Chronos.label(), &upper)?;

        Ok(ForecastOutput {
            model_name: Model::Chronos.label().to_string(),
            horizon_steps: CHRONOS_HORIZON,
            predictions,
            lower: Some(lower),
            upper: Some(upper),
            lookback: CHRONOS_CONTEXT,
        })
    }

    /// Classify the last 30 OHLCV bars as BUY / HOLD / SELL.
    ///
    /// Delegates to [`crate::signal::WatchSignalModel`], which owns the model's
    /// documented 30x55 feature contract; this model is not one of the
    /// window-based forecasters above.
    pub fn predict_signal(&self, candles: &[bt_core::Candle]) -> Result<SignalOutput, ModelError> {
        // `WatchSignalModel::new` takes a *file* path, not a directory.
        let path = self.models_dir.join(crate::signal::SIGNAL_MODEL_FILE);
        let model = crate::signal::WatchSignalModel::new(&path.to_string_lossy())
            .map_err(|e| ModelError::Runtime(e.to_string()))?;
        let out = model
            .predict_candles(candles)
            .map_err(|e| ModelError::Runtime(e.to_string()))?;
        // `Signal::label` is the canonical BUY/HOLD/SELL spelling. The per-class
        // probabilities are not exposed by the classifier, only the winning
        // class and its confidence, so the rest stays zero here rather than
        // being invented.
        Ok(SignalOutput {
            signal: out.signal.label().to_string(),
            confidence: out.confidence,
            probabilities: [0.0; 3],
        })
    }

    /// Run a model over a streaming buffer, pulling from history only once the
    /// buffer holds a full window.
    pub fn predict_streaming(
        &self,
        model: Model,
        buffer: &mut SlidingBuffer,
        latest: f64,
    ) -> Result<Option<ForecastOutput>, ModelError> {
        buffer.push(latest as f32);
        if !buffer.is_ready() {
            return Ok(None);
        }
        let mut window = vec![0.0f32; buffer.capacity()];
        if !buffer.copy_window(&mut window) {
            return Ok(None);
        }
        let closes: Vec<f64> = window.iter().map(|v| *v as f64).collect();
        let out = match model {
            Model::DLinear => self.predict_dlinear(&closes)?,
            Model::NHiTS => self.predict_nhits(&closes)?,
            Model::Chronos => self.predict_chronos(&closes)?,
        };
        Ok(Some(out))
    }

    /// Sessions currently resident, for diagnostics in the UI.
    pub fn cached_sessions(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Drop every cached session, releasing the weights.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

fn poisoned<T>(_: T) -> ModelError {
    ModelError::Runtime("ONNX session cache lock poisoned".into())
}

fn ensure_finite(model: &'static str, values: &[f64]) -> Result<(), ModelError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(ModelError::NonFinite(model))
    }
}

/// Normalize a close series to the last-close-anchored model input contract.
///
/// Returns the normalized window, the anchor price, and the window's standard
/// deviation, which the caller needs to map predictions back to prices.
fn normalize_window(closes: &[f64], need: usize) -> Result<(Vec<f32>, f64, f64), ModelError> {
    if closes.len() < need {
        return Err(ModelError::InsufficientHistory {
            needed: need,
            got: closes.len(),
        });
    }
    let tail = &closes[closes.len() - need..];
    let anchor = *tail.last().unwrap_or(&1.0);
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let variance = tail.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / tail.len() as f64;
    let sd = variance.sqrt();
    // A perfectly flat window has zero scale; 1.0 keeps the input finite and
    // makes the anchor a pure passthrough.
    let sd = if sd.is_finite() && sd > 1e-12 {
        sd
    } else {
        1.0
    };

    let window = tail.iter().map(|v| ((v - anchor) / sd) as f32).collect();
    Ok((window, anchor, sd))
}

/// A model this engine can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    DLinear,
    NHiTS,
    Chronos,
}

impl Model {
    pub const ALL: [Model; 3] = [Model::Chronos, Model::NHiTS, Model::DLinear];

    /// Display name, also used in errors and the UI picker.
    pub fn label(&self) -> &'static str {
        match self {
            Model::DLinear => "DLinear",
            Model::NHiTS => "N-HiTS (small)",
            Model::Chronos => "Chronos-Bolt Tiny (int8)",
        }
    }

    /// Points of history the model needs.
    pub fn required_history(&self) -> usize {
        match self {
            Model::DLinear | Model::NHiTS => DLINEAR_LOOKBACK,
            Model::Chronos => CHRONOS_CONTEXT,
        }
    }

    /// Steps the model emits.
    pub fn horizon(&self) -> usize {
        match self {
            Model::DLinear | Model::NHiTS => DLINEAR_HORIZON,
            Model::Chronos => CHRONOS_HORIZON,
        }
    }

    /// True when the model also emits a quantile band.
    pub fn has_quantile_band(&self) -> bool {
        matches!(self, Model::Chronos)
    }

    fn slot(&self) -> Slot {
        match self {
            Model::DLinear => Slot::DLinear,
            Model::NHiTS => Slot::NHits,
            Model::Chronos => Slot::Chronos,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    #[test]
    fn test_normalize_window_anchors_on_last_close() {
        let (window, anchor, sd) = normalize_window(&ramp(32), 32).unwrap();
        assert_eq!(window.len(), 32);
        assert!((anchor - 115.5).abs() < 1e-9, "anchor {anchor}");
        assert!((window[31] - 0.0).abs() < 1e-6, "last point is the anchor");
        assert!(sd > 0.0);
        // Earlier points are below the anchor, so negative.
        assert!(window[0] < 0.0);
    }

    #[test]
    fn test_normalize_window_uses_only_the_tail() {
        let long: Vec<f64> = (0..100).map(|i| 1000.0 + i as f64).collect();
        let (window, anchor, _) = normalize_window(&long, 32).unwrap();
        assert_eq!(window.len(), 32);
        assert!((anchor - 1099.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_window_flat_series_is_finite() {
        let flat = vec![2500.0; 32];
        let (window, anchor, sd) = normalize_window(&flat, 32).unwrap();
        assert_eq!(anchor, 2500.0);
        assert_eq!(sd, 1.0, "zero scale must be guarded");
        assert!(window.iter().all(|v| v.is_finite()));
        assert!(window.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn test_short_history_is_reported_not_panicked() {
        let err = normalize_window(&ramp(10), 32).unwrap_err();
        assert!(matches!(
            err,
            ModelError::InsufficientHistory {
                needed: 32,
                got: 10
            }
        ));
    }

    #[test]
    fn test_missing_model_reports_clearly() {
        let engine = BharatModelEngine::new("definitely/missing/models");
        let err = engine.predict_dlinear(&ramp(40)).unwrap_err();
        assert!(matches!(err, ModelError::ModelNotFound(_)), "{err:?}");
        assert!(!engine.is_available(Model::DLinear));
        assert!(engine.available_models().is_empty());
        assert_eq!(
            engine.cached_sessions(),
            0,
            "nothing loads on a failed call"
        );
    }

    #[test]
    fn test_construction_performs_no_io() {
        // Must not touch the filesystem or the ONNX runtime.
        let engine = BharatModelEngine::new("no/such/dir");
        assert_eq!(engine.models_dir().to_string_lossy(), "no/such/dir");
        assert_eq!(engine.cached_sessions(), 0);
        engine.clear_cache();
    }

    #[test]
    fn test_model_metadata_is_self_consistent() {
        for model in Model::ALL {
            assert!(!model.label().is_empty());
            assert!(model.required_history() >= model.horizon().min(model.required_history()));
            assert!(model.required_history() > 0);
            assert!(model.horizon() > 0);
        }
        assert_eq!(Model::DLinear.required_history(), 32);
        assert_eq!(Model::DLinear.horizon(), 5);
        assert_eq!(Model::Chronos.required_history(), 64);
        assert_eq!(Model::Chronos.horizon(), 64);
        assert!(Model::Chronos.has_quantile_band());
        assert!(!Model::DLinear.has_quantile_band());
    }

    #[test]
    fn test_streaming_returns_none_until_the_window_fills() {
        let engine = BharatModelEngine::new("no/such/dir");
        let mut buffer = SlidingBuffer::new(32);
        for i in 0..31 {
            // The buffer is not full yet, so no inference is attempted and the
            // missing model file is never even looked for.
            let out = engine.predict_streaming(Model::DLinear, &mut buffer, 100.0 + i as f64);
            assert!(out.unwrap().is_none(), "reported ready at sample {i}");
        }
        // The 32nd sample completes the window, and only now is the missing
        // model reported.
        let err = engine
            .predict_streaming(Model::DLinear, &mut buffer, 200.0)
            .unwrap_err();
        assert!(matches!(err, ModelError::ModelNotFound(_)), "{err:?}");
    }

    #[test]
    fn test_chronos_band_indices_are_in_range() {
        // Guards the quantile slicing arithmetic used by predict_chronos.
        let head_len = CHRONOS_QUANTILES * CHRONOS_HORIZON;
        assert_eq!(head_len, 9 * 64);
        for q in 0..CHRONOS_QUANTILES {
            assert!(q * CHRONOS_HORIZON + CHRONOS_HORIZON <= head_len);
        }
    }
}

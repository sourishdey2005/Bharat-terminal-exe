// crates/bt-analytics/src/models/mod.rs
// Author: Sourish Dey

//! On-device inference for the small local models.
//!
//! Everything here runs locally: no network calls and no external service. Most
//! engines go through the pinned ONNX Runtime, with sessions loaded lazily and
//! cached under an LRU cap so the 2 GB memory budget is respected, and each
//! session configured for single-threaded, `Level1`, memory-pattern-free
//! execution.
//!
//! The one exception is [`py_bridge`], which shells out to an embedded CPython
//! (`python_runtime/`) for a statistical cone that is cheap and weightless. It is
//! still fully local, spawned with `CREATE_NO_WINDOW`, and bounded by a timeout.
//!
//! ```no_run
//! use bt_analytics::models::{BharatModelEngine, Model};
//!
//! let engine = BharatModelEngine::with_default_paths();
//! if engine.is_available(Model::DLinear) {
//!     // DLinear needs 32 points of history; a short slice is an error, not a
//!     // panic.
//!     let closes: Vec<f64> = (0..40).map(|i| 2400.0 + i as f64).collect();
//!     let out = engine.predict_dlinear(&closes)?;
//!     println!("{} -> {:?}", out.model_name, out.predictions);
//! }
//! # Ok::<(), bt_analytics::models::ModelError>(())
//! ```

pub mod contracts;
pub mod engine;
pub mod narrative;
pub mod patchtst_engine;
pub mod py_bridge;
pub mod quantile_engine;
pub mod skill;
pub mod sliding_window;
pub mod ttm_engine;

pub use contracts::{
    split_quantiles, CHRONOS_FLAT_LEN, CHRONOS_HORIZON as CONTRACTS_CHRONOS_HORIZON,
    CHRONOS_P10_INDEX, CHRONOS_P50_INDEX, CHRONOS_P90_INDEX, SIGNAL_N_FEATURES,
};
pub use engine::{
    BharatModelEngine, ForecastOutput, Model, ModelError, CHRONOS_CONTEXT, CHRONOS_HORIZON,
    CHRONOS_QUANTILES, DLINEAR_HORIZON, DLINEAR_LOOKBACK, FILE_CHRONOS, FILE_CHRONOS_FP32,
    FILE_DLINEAR, FILE_NHITS, MAX_CACHED_SESSIONS, NHITS_HORIZON, NHITS_LOOKBACK, SIGNAL_WINDOW,
};
pub use narrative::{MarketCommentary, NarrativeEngine};
pub use patchtst_engine::{
    PatchTstEngine, PatchTstError, PatchTstForecastResult, FILE_PATCHTST, PATCHTST_HORIZON,
    PATCHTST_LOOKBACK,
};
pub use py_bridge::{EmbeddedPyEngine, PyBridgeError, PyStatus, PythonForecastResult, PY_MIN_BARS};
pub use quantile_engine::{QuantileConeOutput, QuantileForecaster};
pub use skill::{is_degenerate, ModelSkill, SkillBook, MIN_EDGE, SKILL_FILE};
pub use sliding_window::SlidingBuffer;
pub use ttm_engine::{TtmEngine, TtmForecastResult, FILE_TTM_R2, TTM_CONTEXT, TTM_HORIZON};

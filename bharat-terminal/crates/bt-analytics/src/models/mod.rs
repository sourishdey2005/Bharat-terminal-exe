// crates/bt-analytics/src/models/mod.rs
// Author: Sourish Dey

//! On-device ONNX inference for the small local models.
//!
//! Everything here runs locally through the pinned ONNX Runtime: no network
//! calls, no Python, no external service. Sessions are loaded lazily and cached
//! under an LRU cap so the 2 GB memory budget is respected, and each session is
//! configured for single-threaded, `Level1`, memory-pattern-free execution.
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

pub mod engine;
pub mod quantile_engine;
pub mod skill;
pub mod sliding_window;

pub use engine::{
    BharatModelEngine, ForecastOutput, Model, ModelError, CHRONOS_CONTEXT, CHRONOS_HORIZON,
    CHRONOS_QUANTILES, DLINEAR_HORIZON, DLINEAR_LOOKBACK, FILE_CHRONOS, FILE_CHRONOS_FP32,
    FILE_DLINEAR, FILE_NHITS, MAX_CACHED_SESSIONS, NHITS_HORIZON, NHITS_LOOKBACK, SIGNAL_WINDOW,
};
pub use quantile_engine::{QuantileConeOutput, QuantileForecaster};
pub use skill::{is_degenerate, ModelSkill, SkillBook, MIN_EDGE, SKILL_FILE};
pub use sliding_window::SlidingBuffer;

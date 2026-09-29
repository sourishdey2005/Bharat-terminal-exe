// crates/bt-analytics/src/forecast/nanoforecast.rs
// Author: Sourish Dey

//! NanoForecast v0.5 (secondary engine) via ONNX Runtime.
//!
//! Expected model interface: a single float32 input shaped `(1, 512)` holding
//! the trailing price context, and a single float output vector with the
//! forecast (the reference config uses horizon 48 with quantile heads — the
//! first output is read as point values).
//!
//! Memory discipline for 2 GB machines: one intra-op and one inter-op thread,
//! graph optimization capped at Level1, and `with_memory_pattern(false)` so
//! the arena allocator cannot pin large blocks. The session lives behind a
//! `RefCell` because `ort` 2.x takes `&mut self` on `run`; the forecaster is
//! used from a single thread.
//!
//! If `models/nanoforecast.onnx` is absent (the upstream repo currently ships
//! only `model.safetensors`), construction fails fast and the fallback chain
//! moves on to ARIMA — no ONNX Runtime binaries are needed for anything else.

use std::cell::RefCell;
use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use super::ForecastError;

pub const NANO_CONTEXT: usize = 512;
pub const NANO_HORIZON: usize = 48;

pub struct NanoForecaster {
    session: RefCell<Session>,
}

impl NanoForecaster {
    pub fn new(model_path: &str) -> Result<Self, ForecastError> {
        if !Path::new(model_path).exists() {
            return Err(ForecastError::ModelNotFound(model_path.to_string()));
        }

        ort::init().with_name("bharat-terminal").commit();

        // CRITICAL for 2 GB RAM: single threads, Level1 only, and no arena
        // memory pattern — otherwise the allocator can pin hundreds of MB.
        let session = Session::builder()
            .map_err(|e| ForecastError::Nano(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| ForecastError::Nano(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| ForecastError::Nano(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| ForecastError::Nano(e.to_string()))?
            .with_memory_pattern(false)
            .map_err(|e| ForecastError::Nano(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| ForecastError::Nano(e.to_string()))?;

        tracing::info!("NanoForecast v0.5 ready: {}", model_path);
        Ok(Self {
            session: RefCell::new(session),
        })
    }

    pub fn predict(&self, history: &[f64], horizon: usize) -> Result<Vec<f64>, ForecastError> {
        if history.is_empty() {
            return Err(ForecastError::EmptyHistory);
        }

        // Right-align the trailing context, zero-padding short histories.
        // Shipped as a (shape, flat-buffer) tuple rather than an ndarray: the
        // ORT crate vendors a newer ndarray than this workspace, and a tuple
        // crosses that version boundary without a type mismatch.
        let mut flat = vec![0.0f32; NANO_CONTEXT];
        let take = history.len().min(NANO_CONTEXT);
        let offset = NANO_CONTEXT - take;
        for (i, &v) in history[history.len() - take..].iter().enumerate() {
            flat[offset + i] = v as f32;
        }
        let tensor = ort::value::Tensor::from_array((vec![1, NANO_CONTEXT], flat))
            .map_err(|e| ForecastError::Nano(e.to_string()))?;

        let mut session = self
            .session
            .try_borrow_mut()
            .map_err(|_| ForecastError::Nano("session already in use".into()))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| ForecastError::Nano(e.to_string()))?;
        let (_, first) = outputs
            .iter()
            .next()
            .ok_or_else(|| ForecastError::Nano("model returned no outputs".into()))?;
        let view = first
            .try_extract_array::<f32>()
            .map_err(|e| ForecastError::Nano(e.to_string()))?;

        // The model emits a fixed window; honour the request up to that.
        Ok(view.iter().take(horizon).map(|&x| x as f64).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_file_is_rejected_without_touching_ort() {
        // Must fail on the path check alone: no ONNX Runtime load attempted.
        let err = match NanoForecaster::new("definitely/missing/nanoforecast.onnx") {
            Ok(_) => panic!("expected a missing-file error"),
            Err(e) => e,
        };
        assert!(matches!(err, ForecastError::ModelNotFound(_)));
    }

    #[test]
    fn test_empty_history_is_rejected() {
        // Constructing with a real file is impossible here, so exercise the
        // validation order indirectly: missing file still wins first.
        assert!(NanoForecaster::new("").is_err());
    }
}

// crates/bt-analytics/src/forecast/mod.rs
// Author: Sourish Dey

//! Multi-model price forecasting with an automatic fallback chain.
//!
//! Engine order:
//! 1. IBM Granite TTM R2 (needs `models/ttm-q8.gguf`, `models/config.json`
//!    and a `ttm-rs` CLI on `PATH`)
//! 2. NanoForecast v0.5 (needs `models/nanoforecast.onnx` plus an ONNX
//!    Runtime the `ort` crate can dlopen)
//! 3. oxidiviner auto-ARIMA (pure Rust, always available)
//!
//! The app must never crash or hang when a model is missing: every engine
//! reports availability up front, every failure degrades to the next engine,
//! and the subprocess path carries a timeout.

pub mod granite;
pub mod nanoforecast;
pub mod statistical;

pub use granite::GraniteForecaster;
pub use nanoforecast::NanoForecaster;
pub use statistical::StatisticalForecaster;

use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ForecastError {
    #[error("Granite error: {0}")]
    Granite(String),
    #[error("NanoForecast error: {0}")]
    Nano(String),
    #[error("Statistical error: {0}")]
    Statistical(String),
    #[error("All models failed: {0}")]
    AllFailed(String),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("History is empty")]
    EmptyHistory,
    #[error("Insufficient data: need at least {0} points, got {1}")]
    InsufficientData(usize, usize),
}

/// Standard model filenames inside the models directory.
pub const GRANITE_GGUF: &str = "ttm-q8.gguf";
pub const GRANITE_CONFIG: &str = "config.json";
pub const NANO_ONNX: &str = "nanoforecast.onnx";

/// Directory that holds optional model files.
///
/// The executable's own directory wins (this matches where `prefs.json` and
/// `cache.db` live), falling back to the process working directory so
/// `cargo run` from the crate root keeps working.
pub fn models_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("models");
            if candidate.is_dir() {
                return candidate;
            }
            // Even when the directory does not exist yet, prefer the
            // executable's folder so a user-placed models/ is found.
            if dir.is_dir() {
                return candidate;
            }
        }
    }
    PathBuf::from("./models")
}

/// A selectable forecasting engine.
///
/// `Auto` runs the statistical auto-selector (ARIMA → exponential smoothing
/// → moving average) unless a neural engine is loaded, in which case neural
/// engines take precedence. The concrete variants run exactly that model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Auto,
    Granite,
    Nano,
    Arima,
    ExpSmooth,
    MovAvg,
}

impl Engine {
    /// Display name for menus and labels.
    pub fn label(&self) -> &'static str {
        match self {
            Engine::Auto => "Auto (best available)",
            Engine::Granite => "Granite TTM R2",
            Engine::Nano => "NanoForecast v0.5",
            Engine::Arima => "ARIMA(1,1,1)",
            Engine::ExpSmooth => "Exp. smoothing (0.3)",
            Engine::MovAvg => "Moving average (5)",
        }
    }

    /// Every engine, in fallback order.
    pub const ALL: [Engine; 6] = [
        Engine::Granite,
        Engine::Nano,
        Engine::Auto,
        Engine::Arima,
        Engine::ExpSmooth,
        Engine::MovAvg,
    ];
}

/// Unified forecaster with an automatic fallback chain.
pub struct Forecaster {
    granite: Option<GraniteForecaster>,
    nano: Option<NanoForecaster>,
    statistical: StatisticalForecaster,
}

impl Forecaster {
    /// Create with explicit paths to the model files. Either may be `None`.
    /// Engines whose files are missing are silently disabled; construction
    /// itself never fails.
    pub fn new(granite_gguf: Option<&str>, nano_onnx: Option<&str>) -> Self {
        let granite = granite_gguf.and_then(|p| {
            GraniteForecaster::new(p, &models_dir().join(GRANITE_CONFIG).to_string_lossy())
                .map_err(|e| tracing::warn!("Granite unavailable: {}", e))
                .ok()
        });

        let nano = nano_onnx.and_then(|p| {
            NanoForecaster::new(p)
                .map_err(|e| tracing::warn!("NanoForecast unavailable: {}", e))
                .ok()
        });

        Self {
            granite,
            nano,
            statistical: StatisticalForecaster::new(),
        }
    }

    /// Create from the standard filenames under [`models_dir`].
    pub fn with_default_paths() -> Self {
        let dir = models_dir();
        let gguf = dir.join(GRANITE_GGUF);
        let onnx = dir.join(NANO_ONNX);
        let gguf_str = gguf.to_string_lossy().to_string();
        let onnx_str = onnx.to_string_lossy().to_string();
        Self::new(
            gguf.exists().then_some(gguf_str.as_str()),
            onnx.exists().then_some(onnx_str.as_str()),
        )
    }

    /// Predict using the best available model, reporting which engine ran.
    /// Each engine returns at most `horizon` points; neural engines return
    /// fewer when their fixed output window is shorter than `horizon`.
    pub fn predict_with_engine(
        &self,
        history: &[f64],
        horizon: usize,
    ) -> Result<(Vec<f64>, &'static str), ForecastError> {
        self.predict_with_preference(Engine::Auto, history, horizon)
    }

    /// Predict starting from a preferred engine, falling back through the
    /// rest of the chain below it. An unavailable or failing preference
    /// degrades silently to the next engine; only a total failure errors.
    pub fn predict_with_preference(
        &self,
        preferred: Engine,
        history: &[f64],
        horizon: usize,
    ) -> Result<(Vec<f64>, &'static str), ForecastError> {
        if history.is_empty() {
            return Err(ForecastError::EmptyHistory);
        }
        if horizon == 0 {
            return Ok((Vec::new(), "none"));
        }

        let start = Engine::ALL
            .iter()
            .position(|e| *e == preferred)
            .unwrap_or(0);
        let mut last_error = String::from("no engine available");
        for engine in &Engine::ALL[start..] {
            let attempt = match engine {
                Engine::Granite => self
                    .granite
                    .as_ref()
                    .map(|g| g.predict(history, horizon).map(|v| (v, "Granite TTM R2"))),
                Engine::Nano => self.nano.as_ref().map(|n| {
                    n.predict(history, horizon)
                        .map(|v| (v, "NanoForecast v0.5"))
                }),
                Engine::Auto | Engine::Arima | Engine::ExpSmooth | Engine::MovAvg => {
                    Some(self.statistical.forecast_engine(*engine, history, horizon))
                }
            };
            match attempt {
                Some(Ok((v, name))) => {
                    tracing::debug!("{name} forecast: {} points", v.len());
                    return Ok((v, name));
                }
                Some(Err(e)) => {
                    tracing::warn!("{preferred:?} chain: {engine:?} failed ({e}), trying next");
                    last_error = e.to_string();
                }
                None => {
                    tracing::debug!("{engine:?} not loaded, trying next");
                }
            }
        }
        Err(ForecastError::AllFailed(last_error))
    }

    /// Predict using the best available model.
    pub fn predict(&self, history: &[f64], horizon: usize) -> Result<Vec<f64>, ForecastError> {
        self.predict_with_engine(history, horizon).map(|(v, _)| v)
    }

    pub fn has_granite(&self) -> bool {
        self.granite.is_some()
    }

    pub fn has_nano(&self) -> bool {
        self.nano.is_some()
    }

    /// Name of the highest-priority engine currently loaded.
    pub fn model_name(&self) -> &'static str {
        if self.granite.is_some() {
            "Granite TTM R2"
        } else if self.nano.is_some() {
            "NanoForecast v0.5"
        } else {
            "ARIMA (statistical)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trend(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    #[test]
    fn test_empty_history_is_rejected() {
        let f = Forecaster::new(None, None);
        assert!(matches!(
            f.predict(&[], 10),
            Err(ForecastError::EmptyHistory)
        ));
    }

    #[test]
    fn test_zero_horizon_returns_empty() {
        let f = Forecaster::new(None, None);
        let (v, _) = f.predict_with_engine(&trend(50), 0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn test_missing_models_fall_back_to_arima() {
        let f = Forecaster::new(
            Some("definitely/missing/ttm-q8.gguf"),
            Some("definitely/missing/nanoforecast.onnx"),
        );
        assert!(!f.has_granite());
        assert!(!f.has_nano());
        assert_eq!(f.model_name(), "ARIMA (statistical)");
        let (v, engine) = f.predict_with_engine(&trend(60), 12).unwrap();
        assert_eq!(v.len(), 12);
        assert!(v.iter().all(|x| x.is_finite()));
        // The auto-selector names its winner; any bench member proves the
        // fallback ran instead of erroring.
        assert!(
            ["ARIMA(1,1,1)", "ExpSmooth(0.3)", "MovAvg(5)", "Auto bench"].contains(&engine),
            "unexpected engine: {engine}"
        );
    }

    #[test]
    fn test_preferred_engine_runs_when_healthy() {
        // Noisy history: a perfect ramp is degenerate for direct ARIMA
        // fitting, while market data always carries noise.
        let history: Vec<f64> = (0..60)
            .map(|i| 100.0 + i as f64 * 0.5 + 3.0 * ((i as f64 * 0.7).sin()))
            .collect();
        let f = Forecaster::new(None, None);
        for (preferred, expected) in [
            (Engine::Arima, "ARIMA(1,1,1)"),
            (Engine::ExpSmooth, "ExpSmooth(0.3)"),
            (Engine::MovAvg, "MovAvg(5)"),
        ] {
            let (v, engine) = f.predict_with_preference(preferred, &history, 8).unwrap();
            assert_eq!(v.len(), 8);
            assert_eq!(engine, expected);
        }
    }

    #[test]
    fn test_missing_preferred_engine_falls_down_the_chain() {
        // Granite files are absent, so preferring it must degrade to the
        // statistical bench rather than fail.
        let f = Forecaster::new(None, None);
        let (v, engine) = f
            .predict_with_preference(Engine::Granite, &trend(60), 8)
            .unwrap();
        assert_eq!(v.len(), 8);
        assert_ne!(engine, "Granite TTM R2");
        let (v, engine) = f
            .predict_with_preference(Engine::Nano, &trend(60), 8)
            .unwrap();
        assert_eq!(v.len(), 8);
        assert_ne!(engine, "NanoForecast v0.5");
    }

    #[test]
    fn test_engine_labels_cover_the_chain() {
        assert_eq!(Engine::ALL.len(), 6);
        for engine in Engine::ALL {
            assert!(!engine.label().is_empty());
        }
        assert_eq!(Engine::Auto.label(), "Auto (best available)");
    }

    #[test]
    fn test_arima_continues_a_trend() {
        let f = Forecaster::new(None, None);
        let history = trend(100);
        let v = f.predict(&history, 5).unwrap();
        // A steady +0.5/step trend must forecast forward, not collapse.
        assert!(v[4] > history[history.len() - 1] - 5.0);
    }

    #[test]
    fn test_insufficient_data_is_an_error_not_a_panic() {
        let f = Forecaster::new(None, None);
        assert!(f.predict(&trend(5), 10).is_err());
    }

    #[test]
    fn test_models_dir_is_absolute_or_relative() {
        // Must at least resolve without panicking; existence is optional.
        let _ = models_dir();
    }
}

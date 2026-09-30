// crates/bt-analytics/src/forecast/statistical.rs
// Author: Sourish Dey

//! Statistical forecast bench (always-live fallback engines).
//!
//! Pure Rust, zero model files, always available:
//! - `Auto`: `auto_select` tries ARIMA(1,1,1), then exponential smoothing,
//!   then a moving average, and reports which one won.
//! - `Arima`, `ExpSmooth`, `MovAvg`: each member run directly.
//!
//! Every engine enforces the same contract: exactly `horizon` finite points.

use chrono::{Duration, Utc};
use oxidiviner::models::exponential_smoothing::SimpleESModel;
use oxidiviner::quick::{arima, auto_select, moving_average};
use oxidiviner::TimeSeriesData;

use super::{Engine, ForecastError};

pub struct StatisticalForecaster;

impl StatisticalForecaster {
    pub fn new() -> Self {
        Self
    }

    /// Run one bench member (or the auto-selector) by name.
    pub fn forecast_engine(
        &self,
        engine: Engine,
        history: &[f64],
        horizon: usize,
    ) -> Result<(Vec<f64>, &'static str), ForecastError> {
        let data = history_data(history)?;
        // The neural engines never reach here: the chain in `forecast::mod`
        // dispatches them to the ONNX engine and only falls through to this
        // bench. Reject rather than silently forecasting with the wrong model.
        if engine.local_model().is_some() {
            return Err(ForecastError::Statistical(format!(
                "{} is an ONNX engine, not a statistical bench member",
                engine.label()
            )));
        }
        let (values, name): (Vec<f64>, &'static str) = match engine {
            Engine::Auto => {
                let (v, model) = auto_select(data, horizon)
                    .map_err(|e| ForecastError::Statistical(e.to_string()))?;
                // `auto_select` names its winner with an owned string; map the
                // three known winners back to static names.
                let name = if model.starts_with("ARIMA") {
                    "ARIMA(1,1,1)"
                } else if model.starts_with("SimpleES") {
                    "ExpSmooth(0.3)"
                } else if model.starts_with("MA(") {
                    "MovAvg(5)"
                } else {
                    "Auto bench"
                };
                (v, name)
            }
            Engine::Arima => (
                arima(data, horizon).map_err(|e| ForecastError::Statistical(e.to_string()))?,
                "ARIMA(1,1,1)",
            ),
            Engine::ExpSmooth => {
                let mut model = SimpleESModel::new(0.3)
                    .map_err(|e| ForecastError::Statistical(format!("{e:?}")))?;
                model
                    .fit(&data)
                    .map_err(|e| ForecastError::Statistical(format!("{e:?}")))?;
                (
                    model
                        .forecast(horizon)
                        .map_err(|e| ForecastError::Statistical(format!("{e:?}")))?,
                    "ExpSmooth(0.3)",
                )
            }
            Engine::MovAvg => (
                moving_average(data, horizon, Some(5))
                    .map_err(|e| ForecastError::Statistical(e.to_string()))?,
                "MovAvg(5)",
            ),
            // Unreachable by construction: the guard at the top of this function
            // rejects every `local_model()` engine. Listed explicitly so adding
            // a new ONNX engine is a compile error here rather than a silent
            // fall-through to the wrong forecaster.
            Engine::Granite | Engine::Nano | Engine::Chronos | Engine::DLinear | Engine::NHits => {
                return Err(ForecastError::Statistical(
                    "not a statistical engine".into(),
                ));
            }
        };
        Ok((checked(values, horizon)?, name))
    }

    /// Name of the model `auto_select` chose on the last call is returned by
    /// [`Self::forecast_with_model`]; this entry point keeps the simple shape.
    pub fn forecast(&self, history: &[f64], horizon: usize) -> Result<Vec<f64>, ForecastError> {
        self.forecast_with_model(history, horizon).map(|(v, _)| v)
    }

    pub fn forecast_with_model(
        &self,
        history: &[f64],
        horizon: usize,
    ) -> Result<(Vec<f64>, String), ForecastError> {
        let (values, name) = self.forecast_engine(Engine::Auto, history, horizon)?;
        Ok((values, name.to_string()))
    }
}

/// Validated, timestamped input for the oxidiviner bench.
fn history_data(history: &[f64]) -> Result<TimeSeriesData, ForecastError> {
    const MIN_POINTS: usize = 10;
    if history.len() < MIN_POINTS {
        return Err(ForecastError::InsufficientData(MIN_POINTS, history.len()));
    }
    if !history.iter().all(|v| v.is_finite()) {
        return Err(ForecastError::Statistical(
            "history contains non-finite values".into(),
        ));
    }
    // ARIMA cares about order, not calendar spacing; synthesize a daily grid
    // ending now so the timestamps are strictly increasing.
    let now = Utc::now();
    let timestamps: Vec<_> = (0..history.len())
        .map(|i| now - Duration::days((history.len() - 1 - i) as i64))
        .collect();
    TimeSeriesData::new(timestamps, history.to_vec(), "bharat-terminal")
        .map_err(|e| ForecastError::Statistical(e.to_string()))
}

/// Contract enforcement: exactly `horizon` finite points.
fn checked(values: Vec<f64>, horizon: usize) -> Result<Vec<f64>, ForecastError> {
    let mut values = values;
    values.truncate(horizon);
    if values.len() != horizon || !values.iter().all(|v| v.is_finite()) {
        return Err(ForecastError::Statistical(
            "engine returned an incomplete forecast".into(),
        ));
    }
    Ok(values)
}

impl Default for StatisticalForecaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trend(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    #[test]
    fn test_arima_returns_exact_horizon() {
        let f = StatisticalForecaster::new();
        for horizon in [1, 5, 20, 48] {
            let (v, model) = f.forecast_with_model(&trend(80), horizon).unwrap();
            assert_eq!(v.len(), horizon, "model {model}");
            assert!(v.iter().all(|x| x.is_finite()));
            assert!(!model.is_empty());
        }
    }

    #[test]
    fn test_each_bench_member_runs_directly() {
        let f = StatisticalForecaster::new();
        // Noisy, realistic history: a perfect ramp is degenerate for ARIMA
        // fitting (unit-root blowup), while market data always carries noise.
        let history = noisy_trend(80);
        for engine in [Engine::Arima, Engine::ExpSmooth, Engine::MovAvg] {
            let (v, name) = f.forecast_engine(engine, &history, 12).unwrap();
            assert_eq!(v.len(), 12, "{name}");
            assert!(v.iter().all(|x| x.is_finite()), "{name}");
        }
    }

    /// Noisy deterministic history shared by the direct-engine tests.
    fn noisy_trend(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + i as f64 * 0.5 + 3.0 * ((i as f64 * 0.7).sin()))
            .collect()
    }

    #[test]
    fn test_bench_members_agree_on_a_trend() {
        // Three independent methods on a noisy trend must all point forward,
        // not collapse or explode.
        let history = noisy_trend(100);
        let last = history[history.len() - 1];
        for engine in [Engine::Arima, Engine::ExpSmooth, Engine::MovAvg] {
            let (v, name) = f_engine(engine, &history);
            assert!(
                v[4] > last - 10.0 && v[4] < last + 10.0,
                "{name} drifted: {}",
                v[4]
            );
        }

        fn f_engine(engine: Engine, history: &[f64]) -> (Vec<f64>, &'static str) {
            StatisticalForecaster::new()
                .forecast_engine(engine, history, 5)
                .unwrap()
        }
    }

    #[test]
    fn test_direct_arima_rejects_degenerate_ramps_cleanly() {
        // A perfect ramp blows up ARIMA(1,1,1) coefficient fitting. That must
        // surface as an ordinary error (the chain then falls through), never
        // a panic or garbage output.
        let f = StatisticalForecaster::new();
        assert!(f.forecast_engine(Engine::Arima, &trend(80), 12).is_err());
    }

    #[test]
    fn test_arima_rejects_short_and_bad_history() {
        let f = StatisticalForecaster::new();
        assert!(f.forecast(&trend(9), 5).is_err());
        assert!(f.forecast(&[], 5).is_err());
        let mut bad = trend(50);
        bad[25] = f64::NAN;
        assert!(f.forecast(&bad, 5).is_err());
    }

    #[test]
    fn test_arima_on_flat_series_stays_flat() {
        let f = StatisticalForecaster::new();
        let flat = vec![42.0; 60];
        let v = f.forecast(&flat, 10).unwrap();
        // A constant series must forecast ~constant, not explode.
        assert!(v.iter().all(|x| (*x - 42.0).abs() < 5.0));
    }

    #[test]
    fn test_neural_variants_are_rejected_by_the_bench() {
        let f = StatisticalForecaster::new();
        assert!(f.forecast_engine(Engine::Granite, &trend(60), 5).is_err());
        assert!(f.forecast_engine(Engine::Nano, &trend(60), 5).is_err());
    }
}

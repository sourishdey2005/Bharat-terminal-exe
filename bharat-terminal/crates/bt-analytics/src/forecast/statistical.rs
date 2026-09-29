// crates/bt-analytics/src/forecast/statistical.rs
// Author: Sourish Dey

//! oxidiviner auto-ARIMA (fallback engine).
//!
//! Pure Rust, zero model files, always available. `auto_select` tries
//! ARIMA(1,1,1), then exponential smoothing, then a moving average, and
//! reports which one won. Forecasts exactly `horizon` points of real,
//! finite values.

use chrono::{Duration, Utc};
use oxidiviner::quick::auto_select;
use oxidiviner::TimeSeriesData;

use super::ForecastError;

pub struct StatisticalForecaster;

impl StatisticalForecaster {
    pub fn new() -> Self {
        Self
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
        const MIN_POINTS: usize = 10;
        if history.len() < MIN_POINTS {
            return Err(ForecastError::InsufficientData(MIN_POINTS, history.len()));
        }
        if !history.iter().all(|v| v.is_finite()) {
            return Err(ForecastError::Statistical(
                "history contains non-finite values".into(),
            ));
        }

        // ARIMA cares about order, not calendar spacing; synthesize a daily
        // grid ending now so the timestamps are strictly increasing.
        let now = Utc::now();
        let timestamps: Vec<_> = (0..history.len())
            .map(|i| now - Duration::days((history.len() - 1 - i) as i64))
            .collect();
        let data = TimeSeriesData::new(timestamps, history.to_vec(), "bharat-terminal")
            .map_err(|e| ForecastError::Statistical(e.to_string()))?;
        let (mut forecast, model) =
            auto_select(data, horizon).map_err(|e| ForecastError::Statistical(e.to_string()))?;

        // Contract: exactly `horizon` finite points. `auto_select` honours the
        // period count, but enforce it here so callers never have to care.
        forecast.truncate(horizon);
        if forecast.len() != horizon || !forecast.iter().all(|v| v.is_finite()) {
            return Err(ForecastError::Statistical(
                "engine returned an incomplete forecast".into(),
            ));
        }
        tracing::debug!("ARIMA forecast via {}", model);
        Ok((forecast, model))
    }
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
}

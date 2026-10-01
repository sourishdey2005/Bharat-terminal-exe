// crates/bt-analytics/src/lib.rs
// Author: Sourish Dey

//! bt-analytics: Technical indicators and risk analytics for Bharat Terminal.
//!
//! Provides pure, deterministic computations for:
//! - Trend indicators (SMA, EMA, MACD, ADX, Parabolic SAR)
//! - Momentum oscillators (RSI, Stochastic, Williams %R, ROC, CCI)
//! - Volatility indicators (Bollinger Bands, ATR, Keltner, Donchian)
//! - Volume indicators (OBV, VWAP, CMF)
//! - Candle transformations (Heikin-Ashi, Renko)
//! - Risk metrics (VaR, CVaR, Sharpe, Sortino, Max Drawdown)
//! - Portfolio analytics (Beta, Alpha, Correlation, Efficient Frontier)

pub mod extended;
pub mod alerts;
pub mod forecast;
pub mod indicators;
pub mod models;
pub mod ort_runtime;
pub mod quant_analytics;
pub mod risk;
pub mod sentiment;
pub mod signal;

pub use extended::{
    ad_line, alma, aroon, aroon_osc, coppock, dpo, elder_ray, eom, fib_levels, force_index,
    high_low_band, hull_ma, kama, keltner_width, kst, log_returns, mass_index, mfi, mfv, multi_ema,
    multi_sma, pivot_levels, pvt, realized_vol, stoch_rsi, supertrend, trend_intensity, tsi, ulcer,
    ultimate_osc, vortex, vwap_bands, wma, zscore, PivotLevels,
};
pub use forecast::granite::GraniteForecaster;
pub use forecast::nanoforecast::NanoForecaster;
pub use forecast::statistical::StatisticalForecaster;
pub use forecast::{models_dir, Engine, ForecastError, Forecaster};
pub use indicators::{
    adx, atr, bollinger, cci, cmf, donchian, ema, heikin_ashi, keltner, macd, obv, parabolic_sar,
    renko, roc, rsi, sma, stochastic, vwap, williams_r,
};
pub use models::{
    is_degenerate, BharatModelEngine, ForecastOutput, MarketCommentary, Model, ModelError,
    ModelSkill, NarrativeEngine, QuantileConeOutput, QuantileForecaster, SkillBook, SlidingBuffer,
    TtmEngine, TtmForecastResult,
};
pub use alerts::{AlertEvent, AlertKind, AlertRule, MarketSnapshot, condition_message, evaluate};
pub use sentiment::{label as sentiment_label, score as sentiment_score, score_labeled};
pub use quant_analytics::{
    FftCycleOutput, QuantAnalyticsEngine, QuantError, Regime, RegimeOutput, FFT_MAX_BARS,
    FFT_MIN_BARS, REGIME_MIN_BARS, REGIME_STATES,
};
pub use risk::{
    alpha, beta, calmar, correlation, correlation_matrix, covariance_matrix, cvar, drawdown_series,
    efficient_frontier, information_ratio, kurtosis, max_drawdown, rolling_correlation,
    rolling_max_drawdown, rolling_moments, rolling_sharpe, rolling_sortino, rolling_volatility,
    sharpe, skewness, sortino, treynor, var_historical,
};
pub use signal::{Signal, SignalOutput, WatchSignalModel, SIGNAL_N_FEATURES, SIGNAL_SEQ_LEN};

#[cfg(test)]
mod tests {
    use super::*;
    use bt_core::{Candle, OhlcvSeries};

    fn sample_series() -> OhlcvSeries {
        let candles = (0..50)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.5;
                Candle::new(
                    i as f64,
                    base,
                    base + 2.0,
                    base - 1.0,
                    base + 0.5,
                    1000.0 + i as f64 * 10.0,
                )
            })
            .collect();
        OhlcvSeries::new("TEST", candles)
    }

    #[test]
    fn test_all_indicators_run() {
        let s = sample_series();
        let _ = sma(&s, 10);
        let _ = ema(&s, 10);
        let _ = rsi(&s, 14);
        let _ = macd(&s);
        let _ = bollinger(&s, 20, 2.0);
        let _ = atr(&s, 14);
        let _ = vwap(&s);
        let _ = obv(&s);
        let _ = stochastic(&s, 14, 3);
        let _ = adx(&s, 14);
        let _ = cci(&s, 20);
        let _ = williams_r(&s, 14);
        let _ = roc(&s, 10);
        let _ = cmf(&s, 20);
        let _ = keltner(&s, 20, 2.0);
        let _ = donchian(&s, 20);
        let _ = parabolic_sar(&s, 0.02, 0.02, 0.2);
        let _ = heikin_ashi(&s);
        let _ = renko(&s, 1.0);
    }

    #[test]
    fn test_all_risk_metrics_run() {
        let s = sample_series();
        let returns = s.returns();
        let _ = max_drawdown(&s);
        let _ = drawdown_series(&s);
        let _ = sharpe(&returns, 0.05, 252);
        let _ = sortino(&returns, 0.05, 252);
        let _ = var_historical(&returns, 0.95);
        let _ = cvar(&returns, 0.95);
        let _ = rolling_sharpe(&returns, 20, 0.05, 252);
        let _ = rolling_sortino(&returns, 20, 0.05, 252);
        let _ = rolling_max_drawdown(&s, 20);
        let _ = beta(&returns, &returns);
        let _ = alpha(&returns, &returns, 0.05, 252);
        let _ = correlation(&returns, &returns);
        let _ = correlation_matrix(&[("A".to_string(), returns.clone())]);
        let _ = rolling_correlation(&returns, &returns, 20);
        let _ = covariance_matrix(&[("A".to_string(), returns.clone())]);
        let _ = efficient_frontier(&[0.1, 0.15], &[vec![0.04, 0.01], vec![0.01, 0.06]], 5);
        let _ = information_ratio(&returns, &returns, 252);
        let _ = treynor(&returns, &returns, 0.05, 252);
        let _ = calmar(&s, 252);
        let _ = rolling_volatility(&returns, 20, 252);
        let _ = kurtosis(&returns);
        let _ = skewness(&returns);
        let _ = rolling_moments(&returns, 20);
    }
}

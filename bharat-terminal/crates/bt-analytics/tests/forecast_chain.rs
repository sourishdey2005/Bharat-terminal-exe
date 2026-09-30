// crates/bt-analytics/tests/forecast_chain.rs
// Author: Sourish Dey

//! Verifies the auto fallback chain picks a real model on a real history.
//!
//! The unit tests in `forecast/mod.rs` use synthetic ramps and explicit engine
//! preferences. This checks the thing that actually matters to a user: with the
//! models that ship on disk, `predict_with_engine` on a realistic price series
//! returns numbers, and naming a specific engine runs that engine rather than
//! silently falling through.

use bt_analytics::{Engine, Forecaster};

fn closes(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            2400.0 + 3.5 * t + 18.0 * (t * 0.08).sin() + 6.0 * (t * 0.37).cos()
        })
        .collect()
}

fn engine_available(engine: Engine) -> bool {
    engine.local_model().is_some_and(|m| {
        bt_analytics::models::BharatModelEngine::with_default_paths().is_available(m)
    })
}

#[test]
fn test_auto_chain_produces_a_realistic_forecast() {
    let forecaster = Forecaster::with_default_paths();
    let history = closes(200);

    let (values, engine) = forecaster
        .predict_with_engine(&history, 12)
        .expect("auto chain produced nothing");

    assert!(!values.is_empty(), "auto chain returned an empty forecast");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "auto chain produced non-finite values: {values:?}"
    );
    println!("auto chain selected: {engine} ({} points)", values.len());

    // A forecast that jumps orders of magnitude away from the last close is a
    // broken model, not a volatile market.
    let last = *history.last().unwrap();
    for (i, v) in values.iter().enumerate() {
        assert!(
            (v - last).abs() < last * 0.5,
            "step {i} value {v} is implausible against last close {last}"
        );
    }
}

#[test]
fn test_each_installed_neural_engine_runs_when_requested() {
    let forecaster = Forecaster::with_default_paths();
    let history = closes(200);

    for engine in [Engine::Chronos, Engine::DLinear, Engine::NHits] {
        if !engine_available(engine) {
            println!("{} not installed, skipped", engine.label());
            continue;
        }
        let (values, used) = forecaster
            .predict_with_preference(engine, &history, 10)
            .unwrap_or_else(|e| panic!("{} failed: {e}", engine.label()));

        assert_eq!(
            used,
            engine.label(),
            "requesting {} ran something else",
            engine.label()
        );
        assert!(!values.is_empty());
        assert!(values.iter().all(|v| v.is_finite()));
        println!("{} -> {} points", engine.label(), values.len());
    }
}

#[test]
fn test_chronos_honours_a_long_horizon() {
    // Chronos is the only engine that natively emits 64 steps; a short request
    // must not be padded or truncated into something else.
    if !engine_available(Engine::Chronos) {
        println!("Chronos not installed, skipped");
        return;
    }
    let forecaster = Forecaster::with_default_paths();
    let history = closes(200);
    let (values, used) = forecaster
        .predict_with_preference(Engine::Chronos, &history, 64)
        .expect("chronos failed");
    assert_eq!(used, "Chronos-Bolt Tiny (int8)");
    assert_eq!(values.len(), 64, "Chronos should fill the full horizon");
    assert!(values.iter().all(|v| v.is_finite()));
}

#[test]
fn test_short_history_falls_through_instead_of_erroring() {
    // Chronos needs 64 bars. With fewer, the chain must skip it and still
    // produce a forecast from the statistical bench rather than reporting the
    // missing history as a hard error.
    let forecaster = Forecaster::with_default_paths();
    let short = closes(20);
    let (values, engine) = forecaster
        .predict_with_engine(&short, 6)
        .expect("short history should still forecast");
    assert!(!values.is_empty());
    assert!(values.iter().all(|v| v.is_finite()));
    println!("short history fell through to: {engine}");
}

#[test]
fn test_auto_prefers_a_neural_engine_over_the_bench() {
    // Regression: `Auto` used to resolve to its own index in the chain, which
    // sat below the neural engines, so Auto silently ran the statistical bench
    // even with a working model installed.
    let forecaster = Forecaster::with_default_paths();
    let history = closes(200);
    let (_, engine) = forecaster
        .predict_with_engine(&history, 12)
        .expect("auto chain produced nothing");

    if let Some(model) = [Engine::Chronos, Engine::DLinear, Engine::NHits]
        .into_iter()
        .find(|e| engine_available(*e))
    {
        assert_eq!(
            engine,
            model.label(),
            "Auto should reach the installed neural model, not the bench"
        );
        println!("auto correctly selected {engine}");
    } else {
        println!("no neural model installed; auto used the bench ({engine})");
        assert!(
            matches!(
                engine,
                "ARIMA(1,1,1)" | "ExpSmooth(0.3)" | "MovAvg(5)" | "Auto bench"
            ),
            "unexpected engine with no neural models installed: {engine}"
        );
    }
}

#[test]
fn test_forecast_is_deterministic_across_repeat_runs() {
    // Sessions are cached and reused, so a second identical request must return
    // identical numbers. A difference would mean state leaked between runs.
    let forecaster = Forecaster::with_default_paths();
    let history = closes(200);
    let first = forecaster.predict_with_engine(&history, 10).unwrap();
    let second = forecaster.predict_with_engine(&history, 10).unwrap();
    assert_eq!(first.0, second.0, "repeat forecast differed");
    assert_eq!(first.1, second.1, "repeat forecast used a different engine");
}

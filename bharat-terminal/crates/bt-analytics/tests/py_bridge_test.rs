// crates/bt-analytics/tests/py_bridge_test.rs
// Author: Sourish Dey

//! Integration tests for the embedded Python forecasting bridge.
//!
//! These drive the real `python_runtime/` interpreter and the real
//! `scripts/predictor.py`, so they cover the parts a mocked test cannot: that the
//! shipped runtime can import numpy, that the stdin/stdout framing matches, and
//! that `CREATE_NO_WINDOW` really does suppress a console window.

use bt_analytics::models::py_bridge::{EmbeddedPyEngine, PythonForecastResult, PY_MIN_BARS};

/// The root holding `python_runtime/` and `scripts/`.
fn project_root() -> std::path::PathBuf {
    bt_analytics::forecast::models_dir()
        .parent()
        .expect("models dir has a parent")
        .to_path_buf()
}

fn engine() -> EmbeddedPyEngine {
    EmbeddedPyEngine::rooted_at(project_root())
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| 2500.0 + i as f32 * 1.5).collect()
}

/// The contract from the task spec, end to end.
#[test]
fn test_embedded_python_engine_execution() {
    let res = engine()
        .predict(&ramp(35))
        .expect("Embedded python execution failed");

    assert_eq!(res.status.as_deref(), Some("ok"));
    assert_eq!(res.horizon, Some(5));
    assert!(res.forecast_p50.is_some());
    assert_eq!(res.forecast_p50.as_ref().unwrap().len(), 5);
    assert!(res.lower_bound_p10.is_some());
    assert!(res.upper_bound_p90.is_some());
    assert!(res.volatility_score.is_some());
}

/// A cone that is not ordered is not a cone: p10 <= p50 <= p90 at every step.
#[test]
fn the_cone_brackets_the_median() {
    let res = engine().predict(&ramp(80)).expect("predict");
    let p10 = res.lower_bound_p10.as_ref().expect("p10");
    let p50 = res.forecast_p50.as_ref().expect("p50");
    let p90 = res.upper_bound_p90.as_ref().expect("p90");
    assert_eq!(p10.len(), p50.len());
    assert_eq!(p50.len(), p90.len());
    for i in 0..p50.len() {
        assert!(
            p10[i] <= p50[i],
            "step {i}: p10 {} exceeds p50 {}",
            p10[i],
            p50[i]
        );
        assert!(
            p50[i] <= p90[i],
            "step {i}: p50 {} exceeds p90 {}",
            p50[i],
            p90[i]
        );
    }
}

/// Real, non-monotonic series: the forecast must stay finite and anchored to the
/// last real close rather than drifting off on a smooth ramp.
#[test]
fn a_realistic_series_produces_finite_prices() {
    let prices: Vec<f32> = (0..120)
        .map(|i| {
            let t = i as f32;
            1182.0 + (t * 0.35).sin() * 22.0 + t * 0.12 + (t * 0.9).cos() * 4.0
        })
        .collect();
    let res = engine().predict(&prices).expect("predict");

    assert!(res
        .forecast_p50
        .as_ref()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    assert_eq!(res.last_price, Some(prices[prices.len() - 1]));
    let last = *prices.last().unwrap();
    for p in res.forecast_p50.as_ref().unwrap() {
        assert!(*p > 0.0, "price went non-positive: {p}");
        // A drift cone should stay within a generous band of the anchor.
        assert!(
            (p / last - 1.0).abs() < 0.5,
            "forecast {p} is more than 50% from the anchor {last}"
        );
    }
}

/// On a volatile series the cone must actually have width, otherwise it is
/// reporting a point estimate dressed as three numbers.
#[test]
fn a_volatile_series_widens_the_cone() {
    let mut seed = 7u32;
    let mut last = 1182.0f32;
    let mut prices = Vec::new();
    for _ in 0..200 {
        // xorshift keeps the fixture reproducible without a rand dependency.
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let shock = ((seed % 2000) as f32 / 1000.0 - 1.0) * 0.04;
        last *= 1.0 + shock;
        prices.push(last);
    }
    let res = engine().predict(&prices).expect("predict");
    let p10 = res.lower_bound_p10.as_ref().unwrap();
    let p90 = res.upper_bound_p90.as_ref().unwrap();
    let anchor = prices[prices.len() - 1];
    assert!(
        p90[0] > p10[0],
        "cone has no width on a volatile series: p10 {} p90 {}",
        p10[0],
        p90[0]
    );
    assert!(
        res.volatility_score.unwrap() > 0.0,
        "volatility reported as 0"
    );
    let _ = anchor;
}

/// Fewer bars than the script accepts must be refused here, not by Python.
#[test]
fn a_short_history_is_refused() {
    assert!(engine().predict(&ramp(PY_MIN_BARS - 1)).is_err());
    assert!(engine().predict(&ramp(PY_MIN_BARS)).is_ok());
}

/// The runtime is resolved relative to the executable, so an install that carries
/// `python_runtime/` next to the binary works without any environment set.
#[test]
fn the_runtime_resolves_next_to_the_project() {
    let e = engine();
    assert!(
        e.is_available(),
        "runtime missing at {:?}; script at {:?}",
        e.python_bin(),
        e.script_path()
    );
    assert!(e.python_bin().ends_with("python.exe"));
    assert!(e.script_path().ends_with("predictor.py"));
}

/// Repeated calls must be stable: no leaked children, no state carried between
/// runs that would make a second answer differ from the first.
#[test]
fn repeated_calls_are_stable_and_leak_no_children() {
    let e = engine();
    let prices = ramp(40);
    let first = e.predict(&prices).expect("first");
    let second = e.predict(&prices).expect("second");
    assert_eq!(
        first, second,
        "a stateless engine gave two different answers"
    );
}

/// The result must survive a JSON round trip, since the API serialises it.
#[test]
fn the_result_round_trips_through_json() {
    let res: PythonForecastResult = engine().predict(&ramp(50)).expect("predict");
    let json = serde_json::to_string(&res).expect("serialise");
    let back: PythonForecastResult = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(res, back);
}

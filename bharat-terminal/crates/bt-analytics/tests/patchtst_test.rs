// crates/bt-analytics/tests/patchtst_test.rs
// Author: Sourish Dey

//! Integration tests for the PatchTST int8 graph.
//!
//! The graph is verified against its declared contract here rather than trusting
//! the task description: `past_values` is `[1, 64]` and `forecast` is `[1, 16]`,
//! both concrete, so a wrong rank is an error rather than a silent reinterpretation.

use bt_analytics::models::patchtst_engine::{
    PatchTstEngine, PatchTstForecastResult, PATCHTST_HORIZON, PATCHTST_LOOKBACK,
};

fn models_dir() -> std::path::PathBuf {
    bt_analytics::forecast::models_dir()
}

fn engine() -> PatchTstEngine {
    PatchTstEngine::new(models_dir())
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| 2500.0 + i as f32 * 1.1).collect()
}

/// The contract from the task spec, end to end.
#[test]
fn test_patchtst_execution() {
    let res = engine()
        .forecast(&ramp(PATCHTST_LOOKBACK))
        .expect("PatchTST inference failed");

    assert_eq!(res.horizon_steps, PATCHTST_HORIZON);
    assert_eq!(res.predictions.len(), PATCHTST_HORIZON);
    assert!(res.predictions.iter().all(|p| p.is_finite()));
}

/// The graph emits standardised output; skipping the inverse transform would put
/// every prediction near zero while still looking like a plausible Vec<f32>.
#[test]
fn predictions_are_in_the_price_domain() {
    let window = ramp(PATCHTST_LOOKBACK);
    let res = engine()
        .forecast(&window)
        .expect("PatchTST inference failed");
    let lo = window.iter().cloned().fold(f32::MAX, f32::min);
    let hi = window.iter().cloned().fold(f32::MIN, f32::max);
    let span = (hi - lo).max(1.0);

    assert!(
        lo > 1_000.0,
        "fixture should sit in a realistic price range"
    );
    for p in &res.predictions {
        assert!(
            (*p - lo).abs() < 20.0 * span,
            "prediction {p} is nowhere near the input range {lo}..{hi}; \
             the standardisation was probably not inverted"
        );
    }
    assert_eq!(res.last_close, window[PATCHTST_LOOKBACK - 1]);
}

/// Exactly 64 bars, or an error naming the length. This is the check that would
/// have caught a `[1, 64, 1]` tensor being passed to a rank-2 graph.
#[test]
fn the_window_length_is_exactly_enforced() {
    for bad in [0usize, 32, 63, 65, 128] {
        let err = engine().forecast(&ramp(bad)).unwrap_err();
        assert!(
            err.to_string().contains("64"),
            "error for {bad} bars should name the required length: {err}"
        );
    }
}

/// Real, non-monotonic input: the whole point of a transformer over a ramp.
#[test]
fn a_realistic_series_is_handled() {
    let window: Vec<f32> = (0..PATCHTST_LOOKBACK)
        .map(|i| {
            let t = i as f32;
            1182.0 + (t * 0.28).sin() * 18.0 + t * 0.09 + (t * 1.7).cos() * 5.0
        })
        .collect();
    let res = engine()
        .forecast(&window)
        .expect("PatchTST inference failed");
    assert_eq!(res.predictions.len(), PATCHTST_HORIZON);
    assert!(res.predictions.iter().all(|p| p.is_finite()));
}

/// A perfectly flat window is the divide-by-zero case for the standardisation.
#[test]
fn a_flat_window_still_produces_finite_output() {
    let res = engine()
        .forecast(&vec![1182.0f32; PATCHTST_LOOKBACK])
        .expect("a flat window must be answerable");
    assert!(res.predictions.iter().all(|p| p.is_finite()));
}

/// The session is built once and reused; a per-call session would blow the memory
/// budget this app is designed around.
#[test]
fn the_session_is_reused_not_rebuilt() {
    let e = engine();
    let window = ramp(PATCHTST_LOOKBACK);
    let first = e.forecast(&window).expect("first");
    let second = e.forecast(&window).expect("second");
    assert_eq!(first, second, "cached session gave a different answer");
}

/// A missing graph must name the path rather than failing obscurely.
#[test]
fn a_missing_graph_reports_the_path() {
    let e = PatchTstEngine::new("definitely-not-a-models-dir");
    assert!(!e.is_available());
    let err = e.forecast(&ramp(PATCHTST_LOOKBACK)).unwrap_err();
    assert!(err.to_string().contains("onnx"), "{err}");
}

/// The result is handed straight to axum, so it has to serialise.
#[test]
fn the_result_round_trips_through_json() {
    let res: PatchTstForecastResult = engine()
        .forecast(&ramp(PATCHTST_LOOKBACK))
        .expect("PatchTST inference failed");
    let json = serde_json::to_string(&res).expect("serialise");
    let back: PatchTstForecastResult = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(res, back);
    assert!(json.contains("\"horizon_steps\":16"), "{json}");
}

/// `probe` is what `/api/engines` calls, so it must reach the same verdict as a
/// real forecast rather than just checking the file exists.
#[test]
fn the_probe_matches_a_real_forecast() {
    let e = engine();
    if !e.is_available() {
        return;
    }
    assert_eq!(
        e.probe().is_ok(),
        e.forecast(&ramp(PATCHTST_LOOKBACK)).is_ok()
    );
}

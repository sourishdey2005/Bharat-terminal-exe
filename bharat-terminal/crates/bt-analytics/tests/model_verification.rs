// crates/bt-analytics/tests/model_verification.rs
// Author: Sourish Dey

//! End-to-end verification of every locally shipped ONNX model.
//!
//! These tests actually load weights and run inference. They are the check that
//! a model file on disk is real, matches its declared I/O contract, and returns
//! usable numbers, rather than a file that merely exists.
//!
//! They need the pinned ONNX Runtime. In CI and on a developer machine that
//! lives at `native/onnxruntime.dll`, which `ort_runtime::ensure_initialized`
//! finds by walking up from the test binary. When no runtime is present the
//! tests report that and pass, so `cargo test` stays hermetic on machines that
//! have not run the installer build, but a machine that *does* have the runtime
//! gets a real end-to-end check.

use std::path::PathBuf;

use bt_analytics::models::{
    BharatModelEngine, Model, CHRONOS_HORIZON, CHRONOS_QUANTILES, DLINEAR_HORIZON,
};

/// Locate `models/` the same way the app does.
fn models_dir() -> PathBuf {
    bt_analytics::forecast::models_dir()
}

/// Whether a pinned ONNX Runtime is reachable. Reports and skips when not.
fn runtime_available() -> bool {
    match bt_analytics::ort_runtime::ensure_initialized() {
        Ok(path) => {
            println!("using ONNX Runtime at {}", path.display());
            true
        }
        Err(e) => {
            println!("skipping ONNX tests: {e}");
            false
        }
    }
}

/// A realistic daily close series: trending with noise, like real NSE data.
fn closes(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            2400.0 + 3.5 * t + 18.0 * (t * 0.08).sin() + 6.0 * (t * 0.37).cos()
        })
        .collect()
}

/// Models are only tested when their file is actually present.
fn require_model(engine: &BharatModelEngine, model: Model) -> bool {
    if engine.is_available(model) {
        true
    } else {
        println!("{}: not installed, skipped", model.label());
        false
    }
}

#[test]
fn test_all_shipped_forecasting_models_load_and_predict() {
    if !runtime_available() {
        return;
    }
    let engine = BharatModelEngine::with_default_paths();
    let history = closes(200);

    let mut ran = 0;

    if require_model(&engine, Model::DLinear) {
        let out = engine
            .predict_dlinear(&history)
            .expect("DLinear inference failed");
        assert_eq!(out.model_name, "DLinear");
        assert_eq!(out.predictions.len(), DLINEAR_HORIZON, "32 -> 5 contract");
        assert_eq!(out.lookback, 32);
        assert!(
            out.predictions.iter().all(|v| v.is_finite()),
            "non-finite forecast"
        );
        // A z-scored model fed a ~2500 price series must return prices in the
        // same neighbourhood, not near-zero normalized values.
        let last = *history.last().unwrap();
        for (i, p) in out.predictions.iter().enumerate() {
            assert!(
                (p - last).abs() < last * 0.25,
                "step {i} forecast {p} is not near the last close {last}"
            );
        }
        println!("DLinear: {:?}", out.predictions);
        ran += 1;
    }

    if require_model(&engine, Model::NHiTS) {
        let out = engine
            .predict_nhits(&history)
            .expect("N-HiTS inference failed");
        assert_eq!(out.model_name, "N-HiTS (small)");
        assert_eq!(out.predictions.len(), 5, "32 -> 5 contract");
        assert!(out.predictions.iter().all(|v| v.is_finite()));
        println!("N-HiTS: {:?}", out.predictions);
        ran += 1;
    }

    if require_model(&engine, Model::Chronos) {
        let out = engine
            .predict_chronos(&history)
            .expect("Chronos-Bolt inference failed");
        assert_eq!(out.horizon_steps, CHRONOS_HORIZON, "64 -> 64 contract");
        assert_eq!(out.lookback, 64);
        assert_eq!(out.predictions.len(), CHRONOS_HORIZON);
        assert!(out.predictions.iter().all(|v| v.is_finite()));

        // Chronos is a quantile model, so the band must be ordered and must
        // actually bracket the median. An unsorted band means the quantile
        // slicing is wrong, which is exactly the bug this asserts against.
        let lower = out.lower.expect("Chronos must emit a lower band");
        let upper = out.upper.expect("Chronos must emit an upper band");
        assert_eq!(lower.len(), CHRONOS_HORIZON);
        assert_eq!(upper.len(), CHRONOS_HORIZON);
        for i in 0..CHRONOS_HORIZON {
            assert!(
                lower[i] <= out.predictions[i] + 1e-3,
                "step {i}: lower {} above median {}",
                lower[i],
                out.predictions[i]
            );
            assert!(
                out.predictions[i] <= upper[i] + 1e-3,
                "step {i}: median {} above upper {}",
                out.predictions[i],
                upper[i]
            );
        }
        println!(
            "Chronos: median[0]={:.2} band=[{:.2}, {:.2}] ({} quantiles)",
            out.predictions[0], lower[0], upper[0], CHRONOS_QUANTILES
        );
        ran += 1;
    }

    assert!(ran > 0, "no forecasting model was available to test");
}

#[test]
fn test_signal_classifier_runs_on_real_candles() {
    if !runtime_available() {
        return;
    }
    let dir = models_dir();
    if !dir.join("stock_signal_lstm_v1_seed42.onnx").is_file() {
        println!("signal classifier: not installed, skipped");
        return;
    }

    let engine = BharatModelEngine::new(&dir);
    // The classifier needs SIGNAL_MIN_CANDLES (80) before it will score, and
    // builds a 30-bar window, so the fixture has to be comfortably longer than
    // both. Each candle is derived from consecutive closes.
    let prices = closes(140);
    let candles: Vec<bt_core::Candle> = prices
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let (o, c) = (w[0], w[1]);
            bt_core::Candle::new(
                1_700_000_000.0 + i as f64 * 86_400.0,
                o,
                o.max(c) + 4.0,
                o.min(c) - 4.0,
                c,
                1_000_000.0 + i as f64 * 1_000.0,
            )
        })
        .collect();
    assert!(candles.len() >= 80, "fixture too short: {}", candles.len());

    let out = engine
        .predict_signal(&candles)
        .expect("signal classification failed");
    assert!(
        matches!(out.signal.as_str(), "BUY" | "HOLD" | "SELL"),
        "unexpected signal {}",
        out.signal
    );
    assert!(
        (0.0..=1.0).contains(&out.confidence),
        "confidence {} out of range",
        out.confidence
    );
    println!("signal: {} @ {:.3}", out.signal, out.confidence);
}

#[test]
fn test_repeat_inference_is_fast_and_stable() {
    if !runtime_available() {
        return;
    }
    let engine = BharatModelEngine::with_default_paths();
    let history = closes(200);
    if !engine.is_available(Model::DLinear) {
        println!("DLinear: not installed, skipped");
        return;
    }

    let first = engine.predict_dlinear(&history).expect("first run failed");

    // Sessions are cached, so the second call must not re-read 4 KB of weights
    // from disk; with the session resident this is sub-millisecond.
    let started = std::time::Instant::now();
    let mut last = first.clone();
    for _ in 0..20 {
        last = engine.predict_dlinear(&history).expect("repeat run failed");
    }
    let per_call = started.elapsed() / 20;

    assert_eq!(
        first.predictions, last.predictions,
        "cached inference must be deterministic"
    );
    assert!(
        per_call < std::time::Duration::from_millis(50),
        "repeat inference too slow: {per_call:?} per call"
    );
    assert_eq!(engine.cached_sessions(), 1, "session should be cached once");
    println!("cached DLinear inference: {per_call:?} per call");
}

#[test]
fn test_session_cache_stays_bounded() {
    if !runtime_available() {
        return;
    }
    let engine = BharatModelEngine::with_default_paths();
    let history = closes(200);

    for model in Model::ALL {
        if !engine.is_available(model) {
            continue;
        }
        let out = match model {
            Model::DLinear => engine.predict_dlinear(&history).map(|o| o.predictions),
            Model::NHiTS => engine.predict_nhits(&history).map(|o| o.predictions),
            Model::Chronos => engine.predict_chronos(&history).map(|o| o.predictions),
        };
        assert!(out.is_ok(), "{} failed: {:?}", model.label(), out.err());
        assert!(
            engine.cached_sessions() <= bt_analytics::models::MAX_CACHED_SESSIONS,
            "session cache exceeded its cap: {}",
            engine.cached_sessions()
        );
    }
    println!(
        "resident sessions after cycling: {}",
        engine.cached_sessions()
    );
}

#[test]
fn test_short_history_is_reported_not_panicked() {
    if !runtime_available() {
        return;
    }
    let engine = BharatModelEngine::with_default_paths();
    if !engine.is_available(Model::Chronos) {
        return;
    }
    // Chronos needs 64 points; 10 must be a clean error, never a panic.
    let err = engine.predict_chronos(&closes(10)).unwrap_err();
    assert!(
        matches!(
            err,
            bt_analytics::models::ModelError::InsufficientHistory {
                needed: 64,
                got: 10
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn test_flat_history_does_not_produce_garbage() {
    if !runtime_available() {
        return;
    }
    let engine = BharatModelEngine::with_default_paths();
    if !engine.is_available(Model::DLinear) {
        return;
    }
    // A perfectly flat series has zero variance; the normalizer must guard the
    // division rather than emit NaN or infinity.
    let flat = vec![2500.0f64; 64];
    let out = engine.predict_dlinear(&flat).expect("flat input failed");
    assert!(out.predictions.iter().all(|v| v.is_finite()));
    for p in &out.predictions {
        assert!(
            (p - 2500.0).abs() < 1.0,
            "flat series should forecast ~2500, got {p}"
        );
    }
}

#[test]
fn test_streaming_buffer_produces_the_same_forecast_as_a_slice() {
    use bt_analytics::models::SlidingBuffer;

    if !runtime_available() {
        return;
    }
    let engine = BharatModelEngine::with_default_paths();
    if !engine.is_available(Model::DLinear) {
        return;
    }

    let history = closes(80);
    let direct = engine.predict_dlinear(&history).expect("direct run failed");

    let mut buffer = SlidingBuffer::new(32);
    let mut last = None;
    for (i, price) in history.iter().enumerate() {
        last = engine
            .predict_streaming(Model::DLinear, &mut buffer, *price)
            .expect("streaming run failed");
        if i < 31 {
            assert!(last.is_none(), "buffer reported ready too early at {i}");
        }
    }

    let streamed = last.expect("buffer never became ready").predictions;
    assert_eq!(
        streamed.len(),
        direct.predictions.len(),
        "streaming and slice paths must agree in length"
    );
    // The buffer stores f32, so the two paths agree to float32 precision rather
    // than bit-for-bit. A real disagreement would be orders of magnitude larger.
    for (a, b) in streamed.iter().zip(direct.predictions.iter()) {
        assert!(
            (a - b).abs() <= 1e-3 * a.abs().max(1.0),
            "streaming {a} and slice {b} disagree"
        );
    }
}

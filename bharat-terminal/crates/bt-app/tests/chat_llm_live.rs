// crates/bt-app/tests/chat_llm_live.rs
// Author: Sourish Dey

//! Live SmolLM2 inference checks. `#[ignore]`d: they need the 105MB GGUF +
//! tokenizer.json in models/ and take seconds per run. Run explicitly:
//! `cargo test -p bt-app --test chat_llm_live -- --ignored --nocapture`.

// Both engine modules are included, because `ChatModel` in chat_llm refers to
// `crate::chat_qwen`. Including only one would leave that path unresolved.
#[path = "../src/chat_llm.rs"]
mod chat_llm;
#[path = "../src/chat_qwen.rs"]
mod chat_qwen;

use chat_llm::{ChatEngineStatus, SmolLM2Engine};
use std::time::Instant;

fn models_dir() -> std::path::PathBuf {
    // Integration tests run with CWD = crate root; the models live two levels
    // up at the workspace root. Walk up until the GGUF is found instead of
    // hardcoding depth.
    let mut dir = std::env::current_dir().expect("cwd");
    loop {
        if dir
            .join("models/SmolLM2-135M-Instruct.Q4_K_M.gguf")
            .is_file()
        {
            return dir.join("models");
        }
        if !dir.pop() {
            return std::path::PathBuf::from("models");
        }
    }
}

fn gguf() -> String {
    models_dir()
        .join("SmolLM2-135M-Instruct.Q4_K_M.gguf")
        .to_string_lossy()
        .into_owned()
}

fn tokenizer() -> String {
    models_dir()
        .join("tokenizer.json")
        .to_string_lossy()
        .into_owned()
}

#[test]
fn engine_status_badges_are_distinct() {
    // Guards the badge match in draw_chat: collapsing two states would show
    // the wrong mode in the window.
    assert_ne!(ChatEngineStatus::Ready, ChatEngineStatus::Unavailable);
    assert_ne!(ChatEngineStatus::Loading, ChatEngineStatus::Unloaded);
}

#[test]
#[ignore = "needs models/*.gguf + tokenizer.json, takes seconds"]
fn live_model_loads() {
    if !std::path::Path::new(&gguf()).is_file() {
        eprintln!("SKIP: GGUF absent");
        return;
    }
    let t = Instant::now();
    let engine = SmolLM2Engine::load(&gguf(), &tokenizer()).expect("load");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());
    assert!(engine.is_loaded());
}

#[test]
#[ignore = "needs models/*.gguf + tokenizer.json, takes seconds"]
fn live_finance_prompt_latency() {
    if !std::path::Path::new(&gguf()).is_file() {
        eprintln!("SKIP: GGUF absent");
        return;
    }
    let mut engine = SmolLM2Engine::load(&gguf(), &tokenizer()).expect("load");
    let prompt = SmolLM2Engine::build_contextual_prompt(
        "Is RELIANCE looking strong this week?",
        "RELIANCE.NS",
        2950.75,
        Some(&[2955.0, 2961.2, 2958.4, 2970.1]),
        Some(62.5),
    );
    let t = Instant::now();
    let out = engine.generate(&prompt, 96).expect("generate");
    let secs = t.elapsed().as_secs_f32();
    eprintln!("finance prompt (96 max) in {secs:.1}s: {out:?}");
    assert!(!out.is_empty(), "empty generation");
}

#[test]
#[ignore = "needs models/*.gguf + tokenizer.json, takes seconds"]
fn live_generates_a_coherent_reply() {
    if !std::path::Path::new(&gguf()).is_file() {
        eprintln!("SKIP: GGUF absent");
        return;
    }
    let mut engine = SmolLM2Engine::load(&gguf(), &tokenizer()).expect("load");
    let prompt = SmolLM2Engine::build_contextual_prompt(
        "What is 2+2? Reply with just the number.",
        "TEST.NS",
        100.0,
        None,
        None,
    );
    let t = Instant::now();
    let out = engine.generate(&prompt, 32).expect("generate");
    let secs = t.elapsed().as_secs_f32();
    eprintln!("generated 32 tokens max in {secs:.1}s: {out:?}");
    assert!(!out.is_empty(), "empty generation");
    assert!(out.len() < 2000, "runaway generation: {out:?}");
}

// crates/bt-analytics/src/signal/mod.rs
// Author: Sourish Dey

//! Trading-signal models (BUY/HOLD/SELL classifiers).
//!
//! Currently hosts the WatchSignal LSTM. Like the forecast engines, loading
//! is fallible and every failure is a plain error — never a panic.

pub mod watchsignal;

pub use watchsignal::{Signal, SignalOutput, WatchSignalModel};
pub use watchsignal::{SIGNAL_N_FEATURES, SIGNAL_SEQ_LEN};

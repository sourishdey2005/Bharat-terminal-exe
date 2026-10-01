// crates/bt-app/src/chat_llm.rs
// Author: Sourish Dey

//! Pure-Rust SmolLM2 inference via Candle — no C++ toolchain required.
//!
//! Wraps `SmolLM2-135M-Instruct.Q4_K_M.gguf` (≈105 MB on disk, ≈140 MB
//! resident) behind a narrow chat API. The engine loads lazily on first chat
//! use, never at startup, and inference always runs on a worker thread — a
//! 135M model needs seconds per reply on CPU, which must never block the UI.
//!
//! Generation uses the KV cache properly: one prefill forward over the prompt
//! at position 0, then single-token forwards with an advancing position. (A
//! common mistake is re-feeding the whole growing sequence at position 0 every
//! step, which recomputes everything and mis-indexes the cache.)

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::instrument;

const DEFAULT_MAX_TOKENS: usize = 96;
const DEFAULT_TEMPERATURE: f64 = 0.7;
const DEFAULT_TOP_P: f64 = 0.9;
const SEED: u64 = 42;

/// Lifecycle of the chat backend, shown as a badge in the chat window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatEngineStatus {
    /// Never attempted (chat not opened yet).
    Unloaded,
    /// Load running on a worker thread.
    Loading,
    /// SmolLM2 answering.
    Ready,
    /// Load failed or files absent; intent matching answers instead.
    Unavailable,
}

/// SmolLM2-135M-Instruct wrapped for single-turn chat inference.
pub struct SmolLM2Engine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    loaded: bool,
}

impl std::fmt::Debug for SmolLM2Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmolLM2Engine")
            .field("loaded", &self.loaded)
            .finish_non_exhaustive()
    }
}

impl SmolLM2Engine {
    /// Load the GGUF model and tokenizer from disk.
    ///
    /// `gguf_path` should point to `SmolLM2-135M-Instruct.Q4_K_M.gguf`;
    /// `tokenizer_path` to the matching `tokenizer.json`
    /// (see `scripts/download-tokenizer.ps1`).
    #[instrument(skip_all, fields(gguf = %gguf_path, tokenizer = %tokenizer_path))]
    pub fn load(gguf_path: &str, tokenizer_path: &str) -> Result<Self, String> {
        if !Path::new(gguf_path).exists() {
            return Err(format!("GGUF not found: {gguf_path}"));
        }
        if !Path::new(tokenizer_path).exists() {
            return Err(format!("Tokenizer not found: {tokenizer_path}"));
        }

        let device = Device::Cpu;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer load: {e}"))?;

        let mut file = std::fs::File::open(gguf_path).map_err(|e| format!("GGUF open: {e}"))?;

        let content =
            gguf_file::Content::read(&mut file).map_err(|e| format!("GGUF parse: {e}"))?;

        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| format!("Model init: {e}"))?;

        tracing::info!("SmolLM2 loaded: {gguf_path} (device=CPU, quant=Q4_K_M)");

        Ok(Self {
            model,
            tokenizer,
            device,
            loaded: true,
        })
    }

    /// Whether the engine is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Generate a completion for a single-turn instruct prompt.
    ///
    /// `max_tokens == 0` selects [`DEFAULT_MAX_TOKENS`]. Stops early on the
    /// instruct end marker or EOS; the cap is a backstop, not the plan.
    #[instrument(skip(self, prompt), fields(prompt_len = prompt.len(), max_tokens))]
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        if !self.loaded {
            return Err("Engine not loaded".into());
        }

        let encoded = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("Tokenize: {e}"))?;
        let mut tokens = encoded.get_ids().to_vec();
        if tokens.is_empty() {
            return Err("Empty token stream".into());
        }

        let mut logits_processor =
            LogitsProcessor::new(SEED, Some(DEFAULT_TEMPERATURE), Some(DEFAULT_TOP_P));

        // Prefill: one forward over the whole prompt at position 0, sampling
        // from the final position's logits.
        let mut pos = 0usize;
        let mut next_token = self.forward_sample(&tokens, pos, &mut logits_processor)?;
        pos += tokens.len();
        tokens.push(next_token);

        let mut output = String::new();
        let max = if max_tokens == 0 {
            DEFAULT_MAX_TOKENS
        } else {
            max_tokens
        };
        // The prefill already produced one token, so at most max-1 more.
        if self.push_piece(next_token, &mut output)? {
            return Ok(output.trim().to_string());
        }
        for _ in 1..max {
            // Decode step: only the latest token enters; the cache holds the rest.
            next_token = self.forward_sample(
                std::slice::from_ref(&next_token),
                pos,
                &mut logits_processor,
            )?;
            pos += 1;
            tokens.push(next_token);
            if self.push_piece(next_token, &mut output)? {
                break;
            }
        }

        Ok(output.trim().to_string())
    }

    /// Forward `tokens` starting at `pos`, return the sampled next token.
    fn forward_sample(
        &mut self,
        tokens: &[u32],
        pos: usize,
        logits_processor: &mut LogitsProcessor,
    ) -> Result<u32, String> {
        let input = Tensor::new(tokens, &self.device)
            .map_err(|e| format!("Tensor: {e}"))?
            .unsqueeze(0)
            .map_err(|e| format!("Unsqueeze: {e}"))?;
        let logits = self
            .model
            .forward(&input, pos)
            .map_err(|e| format!("Forward at pos {pos}: {e}"))?
            .squeeze(0)
            .map_err(|e| format!("Squeeze: {e}"))?;
        logits_processor
            .sample(&logits)
            .map_err(|e| format!("Sample: {e}"))
    }

    /// Append one token's text. Returns true on a stop marker.
    ///
    /// SmolLM2 uses byte-level BPE: `Ġ` marks a word boundary, `Ċ` a newline.
    fn push_piece(&self, token: u32, output: &mut String) -> Result<bool, String> {
        match self.tokenizer.id_to_token(token) {
            None => Err(format!("Unknown token id: {token}")),
            Some(piece) => {
                if piece == "</s>" || piece == "<|im_end|>" {
                    return Ok(true);
                }
                output.push_str(&piece.replace('Ġ', " ").replace('Ċ', "\n"));
                Ok(false)
            }
        }
    }

    /// Build a chat prompt with context from the currently selected symbol.
    ///
    /// This preserves the context-aware behavior the intent-matching chat had:
    /// whatever the app already computed (price, forecast, RSI) goes into the
    /// system block so the model reasons over measured numbers, not vibes.
    pub fn build_contextual_prompt(
        user_question: &str,
        symbol: &str,
        last_price: f64,
        forecast: Option<&[f64]>,
        rsi: Option<f64>,
    ) -> String {
        let mut ctx = format!(
            "You are a financial assistant inside Bharat Terminal, an offline market analysis tool. \
             The user is viewing {symbol}. Last price: \u{20B9}{last_price:.2}. "
        );

        if let Some(f) = forecast {
            if !f.is_empty() {
                if let Some(&last) = f.last() {
                    if last_price.abs() > 1e-12 {
                        let change = (last - last_price) / last_price * 100.0;
                        ctx.push_str(&format!(
                            "Chronos-Bolt forecast {} bars ahead: ₹{last:.2} ({change:+.2}%). ",
                            f.len(),
                            change = change
                        ));
                    }
                }
            }
        }

        if let Some(r) = rsi {
            ctx.push_str(&format!("Current RSI(14): {r:.1}. "));
        }

        ctx.push_str("Answer concisely. Do not give buy/sell advice.\n");

        format!(
            "<|im_start|>system\n{ctx}<|im_end|>\n<|im_start|>user\n{user_question}<|im_end|>\n<|im_start|>assistant\n"
        )
    }
}

/// Convenience: attempt to load from the standard model directory.
///
/// Returns `None` (with a warning, never a panic) when either file is absent
/// or the load fails — the caller falls back to intent matching.
pub fn try_load_default() -> Option<SmolLM2Engine> {
    // Same resolution the ONNX forecasters use, so the LLM is found in exactly
    // the layouts the rest of the app already handles: installed folder, `cargo
    // run` from the project root, and `target/release/` builds.
    let dir = bt_analytics::forecast::models_dir();
    let gguf = dir.join("SmolLM2-135M-Instruct.Q4_K_M.gguf");
    let tok = dir.join("tokenizer.json");
    if !gguf.is_file() || !tok.is_file() {
        tracing::warn!(
            "SmolLM2 or tokenizer missing under {} — chat falls back to intent matching",
            dir.display()
        );
        return None;
    }
    match SmolLM2Engine::load(&gguf.to_string_lossy(), &tok.to_string_lossy()) {
        Ok(e) => Some(e),
        Err(err) => {
            tracing::warn!("SmolLM2 load failed: {err} — intent fallback active");
            None
        }
    }
}

/// Either local model, behind one interface.
///
/// The chat window holds exactly one of these. Making the choice a single enum
/// rather than two parallel `Option`s is what keeps the badge honest: there is
/// only ever one engine field, so the label cannot drift from the model that is
/// actually answering.
pub enum ChatModel {
    /// SmolLM2-135M — fast, small, the default when Qwen is not installed.
    SmolLM2(SmolLM2Engine),
    /// Qwen2.5-1.5B — better answers, roughly an order of magnitude slower.
    Qwen(crate::chat_qwen::QwenEngine),
}

impl ChatModel {
    /// Name shown in the chat badge.
    pub fn badge(&self) -> &'static str {
        match self {
            ChatModel::SmolLM2(_) => "SmolLM2-135M",
            ChatModel::Qwen(_) => "Qwen2.5-1.5B",
        }
    }

    /// Whether the weights are resident.
    pub fn is_loaded(&self) -> bool {
        match self {
            ChatModel::SmolLM2(e) => e.is_loaded(),
            ChatModel::Qwen(e) => e.is_loaded(),
        }
    }

    /// Generate a reply. Both engines share the same KV-cache discipline and
    /// the same `max_tokens == 0` meaning, so this needs no per-model special
    /// casing at the call site.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        match self {
            ChatModel::SmolLM2(e) => e.generate(prompt, max_tokens),
            ChatModel::Qwen(e) => e.generate(prompt, max_tokens),
        }
    }

    /// Build a context-carrying prompt in whichever dialect this model expects.
    ///
    /// The two differ: SmolLM2 uses bare `system`/`user`/`assistant` roles while
    /// Qwen2.5 expects real ChatML `<|im_start|>` markers, and Qwen will
    /// continue the user's turn rather than answering if they are missing.
    pub fn build_contextual_prompt(
        &self,
        user_question: &str,
        symbol: &str,
        last_price: f64,
        forecast: Option<&[f64]>,
        rsi: Option<f64>,
    ) -> String {
        match self {
            ChatModel::SmolLM2(_) => SmolLM2Engine::build_contextual_prompt(
                user_question, symbol, last_price, forecast, rsi,
            ),
            ChatModel::Qwen(_) => crate::chat_qwen::QwenEngine::build_contextual_prompt(
                user_question, symbol, last_price, forecast, rsi,
            ),
        }
    }
}

impl std::fmt::Debug for ChatModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ChatModel").field(&self.badge()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_reports_missing_files_by_path() {
        let err = SmolLM2Engine::load("no-such-model.gguf", "no-such-tok.json").unwrap_err();
        assert!(err.contains("no-such-model.gguf"), "{err}");
        // Tokenizer checked second, only after a real GGUF exists — so craft
        // the reverse: point at Cargo.toml as a fake GGUF and a missing tok.
        let err = SmolLM2Engine::load("Cargo.toml", "no-such-tok.json").unwrap_err();
        assert!(err.contains("no-such-tok.json"), "{err}");
    }

    #[test]
    fn build_contextual_prompt_is_a_well_formed_instruct_string() {
        let f = vec![101.0, 102.0, 103.0];
        let p = SmolLM2Engine::build_contextual_prompt(
            "What is the trend?",
            "RELIANCE.NS",
            100.0,
            Some(&f),
            Some(62.5),
        );
        assert!(p.contains("<|im_start|>system"), "{p}");
        assert!(p.contains("<|im_start|>user"), "{p}");
        assert!(p.ends_with("<|im_start|>assistant\n"), "{p}");
        assert!(p.contains("RELIANCE.NS"), "{p}");
        assert!(p.contains("62.5"), "{p}");
        // +3% forecast move must appear with its sign.
        assert!(p.contains("+3.00%"), "{p}");
        assert!(p.contains("Do not give buy/sell advice"), "{p}");
    }

    #[test]
    fn build_contextual_prompt_survives_missing_context() {
        let p = SmolLM2Engine::build_contextual_prompt("hi", "X", 0.0, None, None);
        assert!(p.contains("<|im_start|>assistant\n"), "{p}");
        assert!(!p.contains("NaN"), "{p}");
        assert!(!p.contains("inf"), "{p}");
    }

    #[test]
    fn try_load_default_never_panics_without_files() {
        // In CI/test CWD there is no models/ dir: must return None quietly.
        // (If a models dir IS present the load may succeed; either is fine,
        // the contract is "no panic".)
        let _ = try_load_default();
    }
}

// crates/bt-app/src/chat_qwen.rs
// Author: Sourish Dey

//! Qwen2.5-1.5B-Instruct inference via Candle — pure Rust, no C++ toolchain.
//!
//! Wraps `qwen2.5-1.5b-instruct-q4_k_m.gguf` (≈990 MB on disk) behind the same
//! narrow chat API [`crate::chat_llm::SmolLM2Engine`] already exposes, so the
//! chat window can treat either model as a drop-in answerer.
//!
//! Deliberately additive: the 135M SmolLM2 remains the fast default and the
//! intent matcher remains the fallback. This is the "better answer, slower"
//! tier, and the badge in the chat window reports which one is live.
//!
//! Generation reuses the same correct KV-cache discipline as the SmolLM2 path:
//! one prefill forward over the whole prompt at position 0, then single-token
//! forwards with an advancing position. Re-feeding the growing sequence at
//! position 0 every step would recompute everything and mis-index the cache.
//!
//! [`QwenEngine`] is the big sibling of [`SmolLM2Engine`](crate::chat_llm::SmolLM2Engine);
//! both are plain structs holding candle weights plus a tokenizer, and neither
//! is `Clone` — the weights are too large to duplicate.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::instrument;

/// Sampling defaults for the 1.5B model.
///
/// The cap is higher than the 135M path (256 vs 96 tokens) because Qwen2.5 is
/// used for explanations that need a sentence or two, not a one-liner. At
/// roughly 11× the parameters it is also roughly an order of magnitude slower
/// per token, so the UI keeps it off the interactive path.
const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_TEMPERATURE: f64 = 0.7;
const DEFAULT_TOP_P: f64 = 0.9;
const SEED: u64 = 42;

/// Filename this engine looks for inside the resolved `models/` directory.
pub const GGUF_FILENAME: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";
/// Tokenizer filename, shared with the SmolLM2 path.
pub const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Qwen2.5-1.5B-Instruct wrapped for single-turn chat inference.
pub struct QwenEngine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    loaded: bool,
}

impl std::fmt::Debug for QwenEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenEngine")
            .field("loaded", &self.loaded)
            .finish_non_exhaustive()
    }
}

impl QwenEngine {
    /// Load the GGUF model and tokenizer from disk.
    ///
    /// `gguf_path` should point at [`GGUF_FILENAME`]; `tokenizer_path` at the
    /// matching `tokenizer.json` (fetched by `scripts/download-qwen.ps1`).
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

        tracing::info!("Qwen2.5-1.5B loaded: {gguf_path} (device=CPU, quant=Q4_K_M)");

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

    /// Generate a completion for a single-turn ChatML prompt.
    ///
    /// `max_tokens == 0` selects [`DEFAULT_MAX_TOKENS`]. Stops early on the
    /// end-of-turn marker or EOS; the cap is a backstop, not the plan.
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

        // Prefill over the whole prompt at position 0, sampling from the last
        // position's logits.
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
        // The prefill already emitted one token, so at most max-1 more.
        if self.push_piece(next_token, &mut output)? {
            return Ok(output.trim().to_string());
        }
        for _ in 1..max {
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
    /// Qwen2.5 uses byte-level BPE: `Ġ` marks a word boundary, `Ċ` a newline.
    fn push_piece(&self, token: u32, output: &mut String) -> Result<bool, String> {
        match self.tokenizer.id_to_token(token) {
            None => Err(format!("Unknown token id: {token}")),
            Some(piece) => {
                if piece == "<|im_end|>" || piece == "<|endoftext|>" || piece.is_empty() {
                    return Ok(true);
                }
                output.push_str(&piece.replace('Ġ', " ").replace('Ċ', "\n"));
                Ok(false)
            }
        }
    }

    /// Build a ChatML prompt carrying market context.
    ///
    /// Qwen2.5-Instruct was trained on ChatML, so the turn markers are real
    /// tokens (`<|im_start|>` / `<|im_end|>`) and must be present for the model
    /// to answer in the assistant role at all.
    ///
    /// The system block is what keeps this honest: the model is told it explains
    /// measured data and never issues buy/sell advice, and it is handed the
    /// numbers the app actually computed rather than asked to invent them.
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
            if let Some(&last) = f.last() {
                if last_price.abs() > 1e-12 {
                    let change = (last - last_price) / last_price * 100.0;
                    ctx.push_str(&format!(
                        "Chronos-Bolt forecast {} bars ahead: \u{20B9}{last:.2} ({change:+.2}%). ",
                        f.len()
                    ));
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

/// Attempt to load Qwen2.5 from the standard model directory.
///
/// Returns `None` (with a warning, never a panic) when either file is absent or
/// the load fails — the caller keeps using SmolLM2 or intent matching.
pub fn try_load_default() -> Option<QwenEngine> {
    // Same resolution the ONNX forecasters use, so the LLM is found in exactly
    // the layouts the rest of the app already handles.
    let dir = bt_analytics::forecast::models_dir();
    let gguf = dir.join(GGUF_FILENAME);
    let tok = dir.join(TOKENIZER_FILENAME);
    if !gguf.is_file() || !tok.is_file() {
        tracing::info!(
            "Qwen2.5-1.5B not installed under {} — chat stays on SmolLM2 or intent matching",
            dir.display()
        );
        return None;
    }
    match QwenEngine::load(&gguf.to_string_lossy(), &tok.to_string_lossy()) {
        Ok(e) => Some(e),
        Err(err) => {
            tracing::warn!("Qwen2.5 load failed: {err} — falling back to the smaller model");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_reports_missing_files_by_path() {
        let err = QwenEngine::load("no-such-model.gguf", "no-such-tok.json").unwrap_err();
        assert!(err.contains("no-such-model.gguf"), "{err}");
        // Tokenizer is only reached once a real GGUF exists, so point the first
        // argument at a file that does exist and the second at a missing one.
        let err = QwenEngine::load("Cargo.toml", "no-such-tok.json").unwrap_err();
        assert!(err.contains("no-such-tok.json"), "{err}");
    }

    #[test]
    fn build_contextual_prompt_is_well_formed_chatml() {
        let f = vec![101.0, 102.0, 103.0];
        let p = QwenEngine::build_contextual_prompt(
            "What is the trend?",
            "RELIANCE.NS",
            100.0,
            Some(&f),
            Some(62.5),
        );
        // ChatML turn markers are what make the model answer in the assistant
        // role; without them Qwen2.5 continues the user turn instead.
        assert!(p.starts_with("<|im_start|>system"), "{p}");
        assert!(p.contains("<|im_start|>user"), "{p}");
        assert!(p.contains("<|im_start|>assistant"), "{p}");
        assert!(p.ends_with("<|im_start|>assistant\n"), "{p}");
        assert!(p.contains("RELIANCE.NS"), "{p}");
        assert!(p.contains("62.5"), "{p}");
        // +3% forecast move must appear with its sign.
        assert!(p.contains("+3.00%"), "{p}");
        assert!(p.contains("Do not give buy/sell advice"), "{p}");
    }

    #[test]
    fn build_contextual_prompt_survives_missing_context() {
        let p = QwenEngine::build_contextual_prompt("hi", "X", 0.0, None, None);
        assert!(p.ends_with("<|im_start|>assistant\n"), "{p}");
        assert!(!p.contains("NaN"), "{p}");
        assert!(!p.contains("inf"), "{p}");
        // A zero last price must not produce a divide-by-zero percentage.
        assert!(!p.contains("%"), "{p}");
    }

    #[test]
    fn forecast_ignored_when_last_price_is_zero() {
        // Zero price with a non-empty forecast previously produced a
        // NaN/inf percentage that leaked into the prompt.
        let f = vec![101.0];
        let p = QwenEngine::build_contextual_prompt("q", "X", 0.0, Some(&f), None);
        assert!(!p.contains("NaN"), "{p}");
        assert!(!p.contains("inf"), "{p}");
    }

    #[test]
    fn try_load_default_never_panics_without_files() {
        // The contract is "no panic", whether or not models/ happens to hold
        // the GGUF.
        let _ = try_load_default();
    }

    #[test]
    fn filenames_match_the_download_script() {
        // scripts/download-qwen.ps1 writes exactly these two names; if either
        // constant drifts the badge would silently never turn green.
        assert_eq!(GGUF_FILENAME, "qwen2.5-1.5b-instruct-q4_k_m.gguf");
        assert_eq!(TOKENIZER_FILENAME, "tokenizer.json");
    }
}
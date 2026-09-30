// crates/bt-analytics/src/signal/watchsignal.rs
// Author: Sourish Dey

//! WatchSignal LSTM — BUY/HOLD/SELL classifier via ONNX Runtime.
//!
//! Model contract: one float32 input shaped `(1, 30, 55)` — 30 trailing bars
//! × 55 features, row-major — and one float output shaped `(1, 4, 3)`: four
//! horizons × (sell, hold, buy) scores.
//!
//! ## Feature layout (PROVISIONAL)
//!
//! The training feature order was never published, so [`build_features`]
//! constructs a documented 55-feature layout from standard technicals
//! (OHLCV, ratios, lags, RSI/SMA/EMA/MACD/Stochastic/Bollinger/ATR/CMF/ADX
//! and volatility stats, all hand-rolled for guaranteed alignment). Every
//! value is guarded finite. Treat live signals as experimental until the
//! training layout is confirmed — the panel labels them as such.
//!
//! ## Calibration
//!
//! A sibling `<model>_temperature.json` (`{"temperature": T}`) is loaded
//! automatically when present. Outputs that already look like probabilities
//! are used as-is; raw logits go through temperature-scaled softmax.

use std::cell::RefCell;
use std::path::Path;

use bt_core::Candle;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SignalError {
    #[error("signal model error: {0}")]
    Onnx(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("feature count mismatch: expected {0}, got {1}")]
    FeatureMismatch(usize, usize),
    #[error("not enough history: need at least {0} candles, got {1}")]
    InsufficientHistory(usize, usize),
}

/// Class order matches the model's output triple `[sell, hold, buy]` — the
/// discriminants are load-bearing (used to index calibrated probabilities),
/// so they are written out explicitly rather than left implicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Sell = 0,
    Hold = 1,
    Buy = 2,
}

impl Signal {
    pub fn label(&self) -> &'static str {
        match self {
            Signal::Sell => "SELL",
            Signal::Hold => "HOLD",
            Signal::Buy => "BUY",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SignalOutput {
    /// The classified signal for the current bar.
    pub signal: Signal,
    /// Winning-class confidence in [0, 1].
    pub confidence: f32,
}

impl SignalOutput {
    /// Headline signal (the single classification).
    pub fn headline(&self) -> Signal {
        self.signal
    }
}

pub const SIGNAL_SEQ_LEN: usize = 30;
pub const SIGNAL_N_FEATURES: usize = 55;
pub const SIGNAL_N_CLASSES: usize = 3;
/// Model filename inside the models directory.
pub const SIGNAL_MODEL_FILE: &str = "stock_signal_lstm_v1_seed42.onnx";

/// Minimum candles needed to warm up the longest lookback (SMA-50 + margin).
pub const SIGNAL_MIN_CANDLES: usize = 80;

pub struct WatchSignalModel {
    session: RefCell<Session>,
    temperature: f32,
}

impl WatchSignalModel {
    pub fn new(model_path: &str) -> Result<Self, SignalError> {
        if !Path::new(model_path).exists() {
            return Err(SignalError::ModelNotFound(model_path.to_string()));
        }

        // Pinned runtime only: a blind init() could dlopen an incompatible
        // system build and crash natively instead of erroring.
        crate::ort_runtime::ensure_initialized().map_err(SignalError::Onnx)?;

        // Same 2 GB discipline as the forecast ONNX path: one thread each
        // way, Level1 only, no arena memory pattern.
        let session = Session::builder()
            .map_err(|e| SignalError::Onnx(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| SignalError::Onnx(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| SignalError::Onnx(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| SignalError::Onnx(e.to_string()))?
            .with_memory_pattern(false)
            .map_err(|e| SignalError::Onnx(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| SignalError::Onnx(e.to_string()))?;

        let temperature = load_temperature(model_path).unwrap_or(1.0);
        tracing::info!(
            "WatchSignal LSTM ready: {} (temperature {})",
            model_path,
            temperature
        );
        Ok(Self {
            session: RefCell::new(session),
            temperature,
        })
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Run inference on a flat row-major `[30 x 55]` feature buffer.
    pub fn predict(&self, features: &[f32]) -> Result<SignalOutput, SignalError> {
        check_feature_len(features)?;
        if !features.iter().all(|v| v.is_finite()) {
            return Err(SignalError::Onnx(
                "feature buffer contains non-finite values".into(),
            ));
        }

        let tensor = ort::value::Tensor::from_array((
            vec![1, SIGNAL_SEQ_LEN, SIGNAL_N_FEATURES],
            features.to_vec(),
        ))
        .map_err(|e| SignalError::Onnx(e.to_string()))?;
        let mut session = self
            .session
            .try_borrow_mut()
            .map_err(|_| SignalError::Onnx("session already in use".into()))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| SignalError::Onnx(e.to_string()))?;
        let (_, first) = outputs
            .iter()
            .next()
            .ok_or_else(|| SignalError::Onnx("model returned no outputs".into()))?;
        let view = first
            .try_extract_array::<f32>()
            .map_err(|e| SignalError::Onnx(e.to_string()))?;
        let data: Vec<f32> = view.iter().copied().collect();

        // The model emits a single (sell, hold, buy) triple.
        if data.len() < SIGNAL_N_CLASSES {
            return Err(SignalError::Onnx(format!(
                "short output: {} values, need {}",
                data.len(),
                SIGNAL_N_CLASSES
            )));
        }
        let triple = [data[0], data[1], data[2]];
        let probs = calibrate(&triple, self.temperature);
        let signal = argmax_signal(&probs);
        Ok(SignalOutput {
            confidence: probs[signal as usize],
            signal,
        })
    }

    /// End-to-end: build provisional features from candles and classify.
    pub fn predict_candles(&self, candles: &[Candle]) -> Result<SignalOutput, SignalError> {
        let features = build_features(candles)?;
        self.predict(&features)
    }
}

/// Map raw model scores to probabilities.
///
/// Outputs that already look like probabilities (all in [0,1], summing to
/// ~1) pass through untouched; anything else is treated as logits and run
/// through temperature-scaled softmax.
fn calibrate(triple: &[f32; 3], temperature: f32) -> [f32; 3] {
    let in_range = triple.iter().all(|&v| v >= 0.0 && v <= 1.0);
    let sum: f32 = triple.iter().sum();
    if in_range && (sum - 1.0).abs() < 0.05 {
        return *triple;
    }
    let t = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        1.0
    };
    let exps = [
        (triple[0] / t).exp(),
        (triple[1] / t).exp(),
        (triple[2] / t).exp(),
    ];
    let total: f32 = exps.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return [0.0, 1.0, 0.0];
    }
    [exps[0] / total, exps[1] / total, exps[2] / total]
}

/// Winning class with deterministic tie-breaking: an exact tie (or NaN)
/// resolves to Hold rather than flipping a coin or biasing long/short.
fn argmax_signal(probs: &[f32; 3]) -> Signal {
    if !(probs[0].is_finite() && probs[1].is_finite() && probs[2].is_finite()) {
        return Signal::Hold;
    }
    if probs[0] > probs[1] && probs[0] > probs[2] {
        Signal::Sell
    } else if probs[2] > probs[0] && probs[2] > probs[1] {
        Signal::Buy
    } else {
        Signal::Hold
    }
}

/// Sibling `<stem>_temperature.json` (`{"temperature": T}`) if present.
fn load_temperature(model_path: &str) -> Option<f32> {
    let sidecar = model_path
        .strip_suffix(".onnx")
        .map(|s| format!("{s}_temperature.json"))?;
    let text = std::fs::read_to_string(sidecar).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let t = value.get("temperature")?.as_f64()? as f32;
    (t.is_finite() && t > 0.0).then_some(t)
}

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        fallback
    }
}

fn safe_div(num: f64, den: f64, fallback: f64) -> f64 {
    if den.abs() > 1e-12 {
        finite_or(num / den, fallback)
    } else {
        fallback
    }
}

fn sma(values: &[f64], end: usize, period: usize) -> Option<f64> {
    if end + 1 < period {
        return None;
    }
    Some(values[end + 1 - period..=end].iter().sum::<f64>() / period as f64)
}

fn ema_series(values: &[f64], period: usize) -> Vec<Option<f64>> {
    let n = values.len();
    let mut out = vec![None; n];
    if n < period || period == 0 {
        return out;
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut ema: Option<f64> = None;
    for i in 0..n {
        ema = Some(match ema {
            None if i + 1 == period => values[..=i].iter().sum::<f64>() / period as f64,
            None => continue,
            Some(prev) => values[i] * k + prev * (1.0 - k),
        });
        out[i] = ema;
    }
    out
}

fn rsi_wilder(closes: &[f64], end: usize, period: usize) -> Option<f64> {
    if end < period || period == 0 {
        return None;
    }
    let mut gain = 0.0;
    let mut loss = 0.0;
    for i in (end + 1 - period)..=end {
        let d = closes[i] - closes[i - 1];
        if d > 0.0 {
            gain += d;
        } else {
            loss -= d;
        }
    }
    if loss <= 1e-12 {
        return Some(100.0);
    }
    let rs = gain / loss;
    Some(100.0 - 100.0 / (1.0 + rs))
}

/// Build the provisional 30×55 feature matrix, row-major.
///
/// Layout per bar (all ratios guarded, everything finite on return):
/// 0-4 raw O,H,L,C,V · 5 log-return · 6-8 body/range/upper-wick over close ·
/// 9 volume vs SMA20 · 10-11 RSI14/100, RSI7/100 · 12-13 SMA20,EMA12 offsets ·
/// 14-16 MACD line/signal/hist · 17-18 stoch K,D /100 · 19 Williams/100 ·
/// 20 ROC10/100 · 21 CCI20/100 · 22 ATR14/close · 23-25 BB offsets+width ·
/// 26 CMF · 27 ADX/100 · 28 Keltner offset · 29 OBV momentum · 30-31 Donchian
/// offsets · 32 VWAP offset · 33-37 close lags 1-5 · 38-42 volume lags 1-5 ·
/// 43 return volatility · 44 SMA50 offset · 45 EMA26 offset · 46-47 high/low
/// momentum · 48 close-vs-SMA20 in ATRs · 49 RSI centered · 50 K-D spread ·
/// 51 MACD hist · 52 intraday position · 53 gap vs prior close · 54 sign(body).
pub fn build_features(candles: &[Candle]) -> Result<Vec<f32>, SignalError> {
    if candles.len() < SIGNAL_MIN_CANDLES {
        return Err(SignalError::InsufficientHistory(
            SIGNAL_MIN_CANDLES,
            candles.len(),
        ));
    }
    let work: Vec<Candle> = candles[candles.len() - 220.min(candles.len())..].to_vec();
    let n = work.len();
    let closes: Vec<f64> = work.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = work.iter().map(|c| c.volume.max(0.0)).collect();
    let ema12 = ema_series(&closes, 12);
    let ema26 = ema_series(&closes, 26);

    // MACD series + signal line.
    let macd_line: Vec<Option<f64>> = (0..n)
        .map(|i| match (ema12[i], ema26[i]) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        })
        .collect();
    let macd_sig = {
        // EMA-9 over the defined MACD stretch, re-aligned by index.
        let defined: Vec<f64> = macd_line.iter().filter_map(|v| *v).collect();
        let sig_all = ema_series(&defined, 9);
        let mut out = vec![None; n];
        let mut k = 0;
        for i in 0..n {
            if macd_line[i].is_some() {
                out[i] = sig_all[k];
                k += 1;
            }
        }
        out
    };

    // Rolling volatility of log-returns + cumulative OBV/VWAP.
    let mut obv = 0.0;
    let mut cum_tpv = 0.0;
    let mut cum_vol = 0.0;

    let mut rows: Vec<[f64; SIGNAL_N_FEATURES]> = Vec::with_capacity(n);
    for i in 0..n {
        let c = &work[i];
        let prev = if i > 0 { &work[i - 1] } else { &work[i] };
        let close = c.close.max(1e-9);
        let mut f = [0.0f64; SIGNAL_N_FEATURES];

        f[0] = c.open;
        f[1] = c.high;
        f[2] = c.low;
        f[3] = c.close;
        f[4] = c.volume.max(0.0);
        f[5] = finite_or((c.close / prev.close.max(1e-9)).ln(), 0.0);
        f[6] = (c.close - c.open) / close;
        f[7] = (c.high - c.low) / close;
        f[8] = (c.high - c.close.max(c.open)) / close;
        f[9] = safe_div(c.volume.max(0.0), sma(&volumes, i, 20).unwrap_or(0.0), 1.0);
        f[10] = rsi_wilder(&closes, i, 14).map_or(0.0, |v| v / 100.0);
        f[11] = rsi_wilder(&closes, i, 7).map_or(0.0, |v| v / 100.0);
        f[12] = sma(&closes, i, 20).map_or(0.0, |v| v / close - 1.0);
        f[13] = ema12[i].map_or(0.0, |v| v / close - 1.0);
        f[14] = macd_line[i].unwrap_or(0.0);
        f[15] = macd_sig[i].unwrap_or(0.0);
        f[16] = macd_line[i].zip(macd_sig[i]).map_or(0.0, |(a, b)| a - b);

        // Stochastic %K/%D over 14 with %D = SMA3 of %K.
        let (k, d) = {
            let s = i.saturating_sub(13);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for c2 in &work[s..=i] {
                lo = lo.min(c2.low);
                hi = hi.max(c2.high);
            }
            let k = safe_div(c.close - lo, hi - lo, 0.5) * 100.0;
            let mut ks = vec![k];
            for j in 1..3 {
                if i >= j {
                    let s2 = (i - j).saturating_sub(13);
                    let (mut lo2, mut hi2) = (f64::INFINITY, f64::NEG_INFINITY);
                    for c2 in &work[s2..=(i - j)] {
                        lo2 = lo2.min(c2.low);
                        hi2 = hi2.max(c2.high);
                    }
                    ks.push(safe_div(work[i - j].close - lo2, hi2 - lo2, 0.5) * 100.0);
                }
            }
            (k, ks.iter().sum::<f64>() / ks.len() as f64)
        };
        f[17] = k / 100.0;
        f[18] = d / 100.0;
        f[19] = {
            let s = i.saturating_sub(13);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for c2 in &work[s..=i] {
                lo = lo.min(c2.low);
                hi = hi.max(c2.high);
            }
            safe_div(c.close - hi, hi - lo, 0.0)
        };
        f[20] = safe_div(c.close - prev.close, prev.close.max(1e-9), 0.0);

        // CCI-20.
        f[21] = {
            let s = i.saturating_sub(19);
            let tps: Vec<f64> = work[s..=i]
                .iter()
                .map(|c2| (c2.high + c2.low + c2.close) / 3.0)
                .collect();
            let mean = tps.iter().sum::<f64>() / tps.len() as f64;
            let md = tps.iter().map(|v| (v - mean).abs()).sum::<f64>() / tps.len() as f64;
            safe_div(tps[tps.len() - 1] - mean, 0.015 * md.max(1e-12), 0.0) / 100.0
        };

        // ATR-14 over close.
        let mut tr_sum = 0.0;
        let mut tr_n = 0;
        for j in (i.saturating_sub(13))..=i {
            if j == 0 {
                continue;
            }
            let cj = &work[j];
            let pj = &work[j - 1];
            tr_sum += (cj.high - cj.low)
                .max((cj.high - pj.close).abs())
                .max((cj.low - pj.close).abs());
            tr_n += 1;
        }
        let atr = if tr_n > 0 { tr_sum / tr_n as f64 } else { 0.0 };
        f[22] = atr / close;

        // Bollinger-20.
        let (bb_mid, bb_up, bb_lo, bb_w) = sma(&closes, i, 20)
            .map(|mid| {
                let s = i + 1 - 20;
                let var = work[s..=i]
                    .iter()
                    .map(|c2| (c2.close - mid).powi(2))
                    .sum::<f64>()
                    / 20.0;
                let sd = var.sqrt();
                (
                    mid,
                    mid + 2.0 * sd,
                    mid - 2.0 * sd,
                    4.0 * sd / close.max(1e-9),
                )
            })
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
        f[23] = safe_div(bb_up - c.close, close, 0.0);
        f[24] = safe_div(c.close - bb_lo, close, 0.0);
        f[25] = bb_w;

        // CMF-20.
        f[26] = {
            let s = i.saturating_sub(19);
            let (mut mf_num, mut mf_den) = (0.0, 0.0);
            for c2 in &work[s..=i] {
                let mfm = safe_div(
                    (c2.close - c2.low) - (c2.high - c2.close),
                    c2.high - c2.low,
                    0.0,
                );
                mf_num += mfm * c2.volume.max(0.0);
                mf_den += c2.volume.max(0.0);
            }
            safe_div(mf_num, mf_den, 0.0)
        };

        // ADX-14 (Wilder, simplified single-pass form).
        f[27] = {
            let mut adx = 0.0;
            if i >= 28 {
                let (mut spdm, mut smdm, mut str_) = (0.0, 0.0, 0.0);
                for j in (i - 27)..=i {
                    let cj = &work[j];
                    let pj = &work[j - 1];
                    let up = cj.high - pj.high;
                    let dn = pj.low - cj.low;
                    let pdm = if up > dn && up > 0.0 { up } else { 0.0 };
                    let mdm = if dn > up && dn > 0.0 { dn } else { 0.0 };
                    let tr = (cj.high - cj.low)
                        .max((cj.high - pj.close).abs())
                        .max((cj.low - pj.close).abs());
                    if j <= i - 14 {
                        spdm += pdm;
                        smdm += mdm;
                        str_ += tr;
                    } else {
                        spdm = spdm - spdm / 14.0 + pdm;
                        smdm = smdm - smdm / 14.0 + mdm;
                        str_ = str_ - str_ / 14.0 + tr;
                    }
                }
                let dip = safe_div(spdm, str_, 0.0) * 100.0;
                let dim = safe_div(smdm, str_, 0.0) * 100.0;
                adx = safe_div((dip - dim).abs(), dip + dim, 0.0);
            }
            adx
        };

        // Keltner offset (EMA20 ± 2*ATR) and OBV momentum.
        let delta = {
            let d = c.close - prev.close;
            if d > 0.0 {
                c.volume.max(0.0)
            } else if d < 0.0 {
                -c.volume.max(0.0)
            } else {
                0.0
            }
        };
        obv += delta;
        let ema20 = ema_series(&closes, 20)[i];
        f[28] = ema20.map_or(0.0, |e| (e + 2.0 * atr - c.close) / close);

        // Donchian-20 offsets and VWAP offset.
        let s20 = i.saturating_sub(19);
        let (mut dhi, mut dlo) = (f64::NEG_INFINITY, f64::INFINITY);
        for c2 in &work[s20..=i] {
            dhi = dhi.max(c2.high);
            dlo = dlo.min(c2.low);
        }
        f[30] = safe_div(dhi - c.close, close, 0.0);
        f[31] = safe_div(c.close - dlo, close, 0.0);
        let tp = (c.high + c.low + c.close) / 3.0;
        cum_tpv += tp * c.volume.max(0.0);
        cum_vol += c.volume.max(0.0);
        f[32] = safe_div(cum_tpv, cum_vol.max(1e-9), c.close) / close - 1.0;

        // Lags, volatility, trend anchors.
        for k in 1..=5 {
            f[32 + k] = if i >= k {
                safe_div(work[i - k].close, close, 1.0)
            } else {
                1.0
            };
            f[37 + k] = if i >= k {
                safe_div(work[i - k].volume.max(0.0), c.volume.max(0.0) + 1.0, 0.0)
            } else {
                0.0
            };
        }
        f[29] = safe_div(delta, c.volume.max(0.0) + 1.0, 0.0);
        f[43] = {
            let s = i.saturating_sub(19);
            let rets: Vec<f64> = (s + 1..=i)
                .map(|j| finite_or((work[j].close / work[j - 1].close.max(1e-9)).ln(), 0.0))
                .collect();
            let mean = rets.iter().sum::<f64>() / rets.len().max(1) as f64;
            (rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len().max(1) as f64).sqrt()
        };
        f[44] = sma(&closes, i, 50).map_or(0.0, |v| v / close - 1.0);
        f[45] = ema26[i].map_or(0.0, |v| v / close - 1.0);
        f[46] = if i > 0 {
            safe_div(c.high - prev.high, prev.high.max(1e-9), 0.0)
        } else {
            0.0
        };
        f[47] = if i > 0 {
            safe_div(c.low - prev.low, prev.low.max(1e-9), 0.0)
        } else {
            0.0
        };
        f[48] = safe_div(c.close - bb_mid, atr.max(1e-9), 0.0);
        f[49] = (rsi_wilder(&closes, i, 14).unwrap_or(50.0) - 50.0) / 50.0;
        f[50] = k - d;
        f[51] = macd_line[i].zip(macd_sig[i]).map_or(0.0, |(a, b)| a - b);
        f[52] = (c.t % 86_400.0) / 86_400.0;
        f[53] = safe_div(c.open - prev.close, prev.close.max(1e-9), 0.0);
        f[54] = (c.close - c.open).signum();

        // Belt and braces: nothing non-finite may reach the model.
        for v in f.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
            // Clamp absurd magnitudes that only arise from degenerate bars.
            if v.abs() > 1e6 {
                *v = v.signum() * 1e6;
            }
        }
        rows.push(f);
    }

    // Emit exactly the trailing 30 rows, flattened row-major f32.
    let tail = rows.len().saturating_sub(SIGNAL_SEQ_LEN);
    Ok(rows[tail..]
        .iter()
        .flat_map(|r| r.iter().map(|&v| v as f32))
        .collect())
}

/// Validates the flat feature buffer shape before any inference is attempted.
fn check_feature_len(features: &[f32]) -> Result<(), SignalError> {
    let expected = SIGNAL_SEQ_LEN * SIGNAL_N_FEATURES;
    if features.len() != expected {
        return Err(SignalError::FeatureMismatch(expected, features.len()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.5 + 2.0 * ((i as f64 * 0.9).sin());
                Candle::new(
                    i as f64 * 86_400.0,
                    base - 1.0,
                    base + 1.5,
                    base - 1.5,
                    base,
                    1_000_000.0 + i as f64 * 1_000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_signal_labels() {
        assert_eq!(Signal::Buy.label(), "BUY");
        assert_eq!(Signal::Hold.label(), "HOLD");
        assert_eq!(Signal::Sell.label(), "SELL");
    }

    #[test]
    fn test_headline_is_the_single_signal() {
        for signal in [Signal::Buy, Signal::Hold, Signal::Sell] {
            let out = SignalOutput {
                signal,
                confidence: 0.8,
            };
            assert_eq!(out.headline(), signal);
        }
    }

    #[test]
    fn test_calibrate_passes_probabilities_and_softmaxes_logits() {
        // Already probabilities: untouched.
        assert_eq!(calibrate(&[0.1, 0.7, 0.2], 1.07), [0.1, 0.7, 0.2]);
        // Logits: temperature softmax, argmax preserved.
        let p = calibrate(&[2.0, 1.0, 0.5], 1.07);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[0] > p[1] && p[1] > p[2]);
        // Degenerate temperature falls back to plain softmax.
        let q = calibrate(&[2.0, 1.0, 0.5], 0.0);
        assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax_is_deterministic_on_ties_and_nan() {
        assert_eq!(argmax_signal(&[0.2, 0.5, 0.3]), Signal::Hold);
        assert_eq!(argmax_signal(&[0.7, 0.2, 0.1]), Signal::Sell);
        assert_eq!(argmax_signal(&[0.1, 0.2, 0.7]), Signal::Buy);
        assert_eq!(argmax_signal(&[0.4, 0.4, 0.2]), Signal::Hold);
        assert_eq!(argmax_signal(&[f32::NAN, f32::NAN, f32::NAN]), Signal::Hold);
    }

    #[test]
    fn test_build_features_shape_finiteness_and_determinism() {
        let cs = candles(120);
        let a = build_features(&cs).unwrap();
        let b = build_features(&cs).unwrap();
        assert_eq!(a.len(), SIGNAL_SEQ_LEN * SIGNAL_N_FEATURES);
        assert!(a.iter().all(|v| v.is_finite()));
        assert_eq!(a, b, "feature builder must be deterministic");
        // Raw OHLCV anchors of the last bar survive untouched.
        let last = &cs[cs.len() - 1];
        let base = (SIGNAL_SEQ_LEN - 1) * SIGNAL_N_FEATURES;
        assert!((a[base] as f64 - last.open).abs() < 1e-3);
        assert!((a[base + 3] as f64 - last.close).abs() < 1e-3);
    }

    #[test]
    fn test_build_features_rejects_short_history() {
        assert!(build_features(&candles(30)).is_err());
        assert!(build_features(&[]).is_err());
    }

    #[test]
    fn test_missing_model_file_is_rejected() {
        match WatchSignalModel::new("definitely/missing/signal.onnx") {
            Ok(_) => panic!("expected a missing-file error"),
            Err(e) => assert!(matches!(e, SignalError::ModelNotFound(_))),
        }
    }

    /// Live end-to-end run against the real ONNX model. Ignored by default
    /// (needs the model file + onnxruntime.dll); run explicitly:
    /// `cargo test -p bt-analytics -- --ignored watchsignal_live`
    #[test]
    #[ignore]
    fn watchsignal_live_end_to_end() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let model = WatchSignalModel::new(
            root.join("stock_signal_lstm_v1_seed42.onnx")
                .to_str()
                .unwrap(),
        )
        .expect("needs model + onnxruntime.dll");
        let cs = candles(120);
        let out = model.predict_candles(&cs).expect("live inference");
        assert!((0.0..=1.0).contains(&out.confidence));
        // The model is deterministic: same input, same signal.
        let again = model.predict_candles(&cs).expect("live inference");
        assert_eq!(out.signal, again.signal);
    }

    #[test]
    fn test_feature_length_guard_rejects_bad_shapes() {
        assert!(check_feature_len(&vec![0.0f32; SIGNAL_SEQ_LEN * SIGNAL_N_FEATURES]).is_ok());
        assert!(matches!(
            check_feature_len(&vec![0.0f32; 10]),
            Err(SignalError::FeatureMismatch(1650, 10))
        ));
        assert!(check_feature_len(&[]).is_err());
    }
}

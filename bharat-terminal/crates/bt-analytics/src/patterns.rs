// crates/bt-analytics/src/patterns.rs
// Author: Sourish Dey

//! Chart-pattern detection over OHLCV candles.
//!
//! Pure functions, no models, no I/O: each detector scores the most recent
//! bars and returns the matches it finds. Thresholds are deliberately
//! conservative (fewer, higher-quality hits) because a screener that flags
//! everything flags nothing.

use bt_core::Candle;

/// Where a pattern was found and how strong it looks.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternHit {
    /// Human name, e.g. `"Hammer"`.
    pub name: &'static str,
    /// Index of the confirming (last) bar.
    pub bar: usize,
    /// 0..1 confidence from the shape margins.
    pub confidence: f64,
}

fn body(c: &Candle) -> f64 {
    (c.close - c.open).abs()
}

fn range(c: &Candle) -> f64 {
    (c.high - c.low).max(1e-12)
}

/// Bullish hammer on the last bar: small body near the high, long lower wick.
pub fn detect_hammer(candles: &[Candle]) -> Option<PatternHit> {
    let c = candles.last()?;
    let r = range(c);
    let b = body(c);
    let lower_wick = c.open.min(c.close) - c.low;
    let upper_wick = c.high - c.open.max(c.close);
    if b < 0.35 * r && lower_wick > 2.0 * b && upper_wick < 0.25 * r {
        let confidence = ((lower_wick / r - 0.5) * 2.0).clamp(0.0, 1.0);
        return Some(PatternHit {
            name: "Hammer",
            bar: candles.len() - 1,
            confidence,
        });
    }
    None
}

/// Doji on the last bar: open ≈ close relative to the range.
pub fn detect_doji(candles: &[Candle]) -> Option<PatternHit> {
    let c = candles.last()?;
    let r = range(c);
    let b = body(c);
    if b < 0.10 * r {
        return Some(PatternHit {
            name: "Doji",
            bar: candles.len() - 1,
            confidence: (1.0 - b / (0.10 * r)).clamp(0.0, 1.0),
        });
    }
    None
}

/// Bullish or bearish engulfing over the last two bars.
pub fn detect_engulfing(candles: &[Candle]) -> Option<PatternHit> {
    if candles.len() < 2 {
        return None;
    }
    let (prev, cur) = (&candles[candles.len() - 2], &candles[candles.len() - 1]);
    let bar = candles.len() - 1;
    // Bullish: red body fully inside the next green body.
    if prev.close < prev.open && cur.close > cur.open && cur.open <= prev.close && cur.close >= prev.open
    {
        let size = (body(cur) / range(cur)).clamp(0.0, 1.0);
        return Some(PatternHit {
            name: "Bullish Engulfing",
            bar,
            confidence: size,
        });
    }
    // Bearish mirror.
    if prev.close > prev.open && cur.close < cur.open && cur.open >= prev.close && cur.close <= prev.open
    {
        let size = (body(cur) / range(cur)).clamp(0.0, 1.0);
        return Some(PatternHit {
            name: "Bearish Engulfing",
            bar,
            confidence: size,
        });
    }
    None
}

/// Head-and-shoulders top over the last `lookback` closes (default 60).
///
/// Finds three ascending-then-descending peaks where the middle (head) is the
/// highest and the two shoulders sit within `tolerance` of each other.
pub fn detect_head_shoulders(closes: &[f64], lookback: usize, tolerance: f64) -> Option<PatternHit> {
    let n = closes.len().min(lookback.max(30));
    if closes.len() < 30 || n < 30 {
        return None;
    }
    let w = &closes[closes.len() - n..];
    // Local peaks (strictly higher than both neighbours).
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..w.len() - 1 {
        if w[i] > w[i - 1] && w[i] > w[i + 1] {
            peaks.push(i);
        }
    }
    if peaks.len() < 3 {
        return None;
    }
    // Take the highest peak as the head; shoulders are the highest peaks on
    // each side of it.
    let head = *peaks.iter().max_by(|a, b| w[**a].partial_cmp(&w[**b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let left = peaks.iter().filter(|p| **p < head).max();
    let right = peaks.iter().filter(|p| **p > head).min();
    let (Some(&l), Some(&r)) = (left, right) else {
        return None;
    };
    if w[head] <= w[l] || w[head] <= w[r] {
        return None;
    }
    let shoulder_diff = ((w[l] - w[r]).abs() / w[head]).max(0.0);
    if shoulder_diff > tolerance {
        return None;
    }
    Some(PatternHit {
        name: "Head and Shoulders",
        bar: closes.len() - 1,
        confidence: (1.0 - shoulder_diff / tolerance.max(1e-9)).clamp(0.0, 1.0),
    })
}

/// Double top: two peaks within `tolerance`, separated by a trough at least
/// `min_depth` (fraction of price) below the lower peak.
pub fn detect_double_top(
    closes: &[f64],
    lookback: usize,
    tolerance: f64,
    min_depth: f64,
) -> Option<PatternHit> {
    let n = closes.len().min(lookback.max(30));
    if closes.len() < 30 || n < 30 {
        return None;
    }
    let w = &closes[closes.len() - n..];
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..w.len() - 1 {
        if w[i] > w[i - 1] && w[i] > w[i + 1] {
            peaks.push(i);
        }
    }
    if peaks.len() < 2 {
        return None;
    }
    // Highest two peaks, ordered in time.
    let mut sorted = peaks.clone();
    sorted.sort_by(|a, b| w[*b].partial_cmp(&w[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let (mut p1, mut p2) = (sorted[0], sorted[1]);
    if p1 > p2 {
        std::mem::swap(&mut p1, &mut p2);
    }
    let top = w[p1].max(w[p2]);
    if (w[p1] - w[p2]).abs() / top > tolerance {
        return None;
    }
    let trough = w[p1..=p2].iter().cloned().fold(f64::MAX, f64::min);
    let depth = (top.min(w[p1]).min(w[p2]) - trough) / top;
    if depth < min_depth {
        return None;
    }
    Some(PatternHit {
        name: "Double Top",
        bar: closes.len() - 1,
        confidence: (depth / (min_depth * 3.0).max(1e-9)).clamp(0.0, 1.0),
    })
}

/// Run every detector over the latest bars and return all hits.
pub fn detect_all(candles: &[Candle]) -> Vec<PatternHit> {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let mut out = Vec::new();
    for hit in [
        detect_hammer(candles),
        detect_doji(candles),
        detect_engulfing(candles),
        detect_head_shoulders(&closes, 60, 0.03),
        detect_double_top(&closes, 60, 0.02, 0.02),
    ]
    .into_iter()
    .flatten()
    {
        out.push(hit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candle(o: f64, h: f64, l: f64, c: f64) -> Candle {
        Candle::new(0.0, o, h, l, c, 1000.0)
    }

    #[test]
    fn hammer_needs_a_long_lower_wick() {
        // Classic hammer: open 100, close 101, low 90, high 102.
        assert!(detect_hammer(&[candle(100.0, 102.0, 90.0, 101.0)]).is_some());
        // Same body, no wick: not a hammer.
        assert!(detect_hammer(&[candle(100.0, 101.5, 99.5, 101.0)]).is_none());
        assert!(detect_hammer(&[]).is_none());
    }

    #[test]
    fn doji_needs_a_tiny_body() {
        assert!(detect_doji(&[candle(100.0, 105.0, 95.0, 100.1)]).is_some());
        assert!(detect_doji(&[candle(100.0, 105.0, 95.0, 103.0)]).is_none());
    }

    #[test]
    fn engulfing_checks_both_directions() {
        let bull = vec![candle(105.0, 106.0, 99.0, 100.0), candle(99.5, 107.0, 99.0, 106.0)];
        let hit = detect_engulfing(&bull).unwrap();
        assert_eq!(hit.name, "Bullish Engulfing");
        let bear = vec![candle(100.0, 106.0, 99.0, 105.0), candle(105.5, 106.0, 99.0, 100.0)];
        let hit = detect_engulfing(&bear).unwrap();
        assert_eq!(hit.name, "Bearish Engulfing");
        assert!(detect_engulfing(&[candle(100.0, 101.0, 99.0, 100.5)]).is_none());
    }

    fn rising_with_peaks() -> Vec<f64> {
        // Three humps: 105, 120 (head), 106 — head-and-shoulders top.
        let mut v = Vec::new();
        for i in 0..60 {
            let t = i as f64;
            v.push(100.0 + t * 0.1 + 8.0 * (t * 0.35).sin() + 4.0 * (t * 0.13).cos());
        }
        v
    }

    #[test]
    fn head_and_shoulders_finds_three_peaks() {
        // Construct explicitly: left 105 @10, head 120 @30, right 106 @50.
        let mut closes = vec![100.0; 60];
        for (i, v) in [(10, 105.0), (30, 120.0), (50, 106.0)] {
            closes[i] = v;
        }
        // Fill neighbours below so they count as strict peaks.
        for i in [9, 11, 29, 31, 49, 51] {
            closes[i] = 100.0;
        }
        let hit = detect_head_shoulders(&closes, 60, 0.05).unwrap();
        assert_eq!(hit.name, "Head and Shoulders");
        // Shoulders 105 vs 106 differ by <5% of head 120.
        let _ = rising_with_peaks();
    }

    #[test]
    fn head_and_shoulders_rejects_lopsided_shoulders() {
        let mut closes = vec![100.0; 60];
        for (i, v) in [(10, 105.0), (30, 120.0), (50, 80.0)] {
            closes[i] = v;
        }
        for i in [9, 11, 29, 31, 49, 51] {
            closes[i] = 70.0;
        }
        assert!(detect_head_shoulders(&closes, 60, 0.05).is_none());
    }

    #[test]
    fn double_top_needs_two_equal_peaks_and_a_trough() {
        let mut closes = vec![100.0; 60];
        for (i, v) in [(15, 120.0), (40, 121.0)] {
            closes[i] = v;
        }
        for i in [14, 16, 39, 41] {
            closes[i] = 100.0;
        }
        for c in &mut closes[20..35] {
            *c = 105.0;
        }
        let hit = detect_double_top(&closes, 60, 0.02, 0.02).unwrap();
        assert_eq!(hit.name, "Double Top");
    }

    #[test]
    fn short_series_detects_nothing() {
        let closes = vec![100.0; 10];
        assert!(detect_head_shoulders(&closes, 60, 0.05).is_none());
        assert!(detect_double_top(&closes, 60, 0.02, 0.02).is_none());
        // Rising candles with healthy bodies: no doji, no hammer, no engulf.
        let candles: Vec<Candle> = (0..5)
            .map(|i| {
                let o = 100.0 + i as f64;
                candle(o, o + 1.0, o - 1.0, o + 0.5)
            })
            .collect();
        assert!(detect_all(&candles).is_empty());
    }
}

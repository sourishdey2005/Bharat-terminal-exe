// crates/bt-analytics/src/signals.rs
// Author: Sourish Dey

//! Aggregates every signal source into one comparable record.
//!
//! The advisor in [`crate::advisor`] produces a verdict; this produces the
//! *evidence table* that goes with it. Keeping the two apart matters: the
//! Advisor tab shows the verdict and this shows, row by row, which inputs were
//! present and how each was read. A reader who disagrees with the verdict can
//! see exactly which row drove it.
//!
//! Sources that produced nothing are simply absent from `sources` — a missing
//! row means "not measured", which is different from a row reading neutral.

use crate::advisor::{advise, Action, AdvisorInput, AdvisorOutput};

/// One signal source, as read.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalSource {
    /// Display name, e.g. "RSI".
    pub name: String,
    /// The reading, already formatted for display.
    pub value: String,
    /// `-1..=1`, the source's own opinion. `0` means neutral.
    pub lean: f64,
}

/// The composite verdict plus the evidence behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositeSignal {
    pub action: Action,
    pub confidence: f64,
    pub score: i32,
    pub rule_output: AdvisorOutput,
    /// Every source that was present, in a stable order.
    pub sources: Vec<SignalSource>,
}

impl CompositeSignal {
    /// Fraction of the scoring inputs that agreed with the verdict.
    ///
    /// Reported separately from `confidence` because "I am 75% confident" and
    /// "4 of 5 inputs point the same way" are different claims, and a reader
    /// deserves to see both. Returns 0 when nothing was measured.
    pub fn agreement(&self) -> f64 {
        let scored: Vec<&SignalSource> =
            self.sources.iter().filter(|s| s.lean != 0.0).collect();
        if scored.is_empty() {
            return 0.0;
        }
        let bullish = scored.iter().filter(|s| s.lean > 0.0).count();
        let bearish = scored.iter().filter(|s| s.lean < 0.0).count();
        let dominant = bullish.max(bearish) as f64;
        dominant / scored.len() as f64
    }
}

/// Build the composite signal for `input`.
///
/// The verdict is delegated wholesale to [`advise`] — this function does not
/// re-score anything, so the Advisor tab and any future consumer cannot
/// disagree about what the rules concluded.
pub fn composite(input: &AdvisorInput) -> CompositeSignal {
    let rule_output = advise(input);
    let mut sources: Vec<SignalSource> = Vec::new();

    if let Some(rsi) = input.rsi.filter(|v| v.is_finite()) {
        // Lean mirrors the advisor's own RSI thresholds, normalized onto -1..1.
        let lean = if rsi < 30.0 {
            0.8
        } else if rsi < 45.0 {
            0.4
        } else if rsi > 70.0 {
            -0.8
        } else if rsi > 55.0 {
            -0.4
        } else {
            0.0
        };
        let note = match lean {
            l if l > 0.0 => "bullish",
            l if l < 0.0 => "bearish",
            _ => "neutral",
        };
        sources.push(SignalSource {
            name: "RSI".into(),
            value: format!("{rsi:.1} ({note})"),
            lean,
        });
    }

    if let Some(macd) = input.macd_signal.filter(|v| v.is_finite()) {
        sources.push(SignalSource {
            name: "MACD".into(),
            value: format!("{macd:+.3} vs signal"),
            lean: if macd > 0.0 { 0.5 } else { -0.5 },
        });
    }

    if let Some(fc) = input.forecast_change_pct.filter(|v| v.is_finite()) {
        // 5% and beyond is treated as a full-strength opinion.
        let lean = (fc / 5.0).clamp(-1.0, 1.0);
        sources.push(SignalSource {
            name: "Forecast".into(),
            value: format!("{fc:+.2}%"),
            lean,
        });
    }

    if let Some(sig) = &input.watchsignal {
        let upper = sig.to_uppercase();
        let lean = if upper.contains("BUY") {
            0.5
        } else if upper.contains("SELL") {
            -0.5
        } else {
            0.0
        };
        sources.push(SignalSource {
            name: "WatchSignal".into(),
            value: sig.clone(),
            lean,
        });
    }

    if let Some(p) = input.price_vs_sma50.filter(|v| v.is_finite()) {
        // 10% above/below SMA50 is a full-strength trend read.
        let lean = (p / 10.0).clamp(-1.0, 1.0);
        sources.push(SignalSource {
            name: "SMA50 trend".into(),
            value: format!("{p:+.1}%"),
            lean,
        });
    }

    if let Some(vr) = input.volume_ratio.filter(|v| v.is_finite()) {
        // Volume is context: a thin tape qualifies a move rather than
        // predicting one, so its lean is never allowed to exceed a weak value.
        let lean = if vr < 0.5 {
            -0.2
        } else if vr > 1.5 {
            0.2
        } else {
            0.0
        };
        sources.push(SignalSource {
            name: "Volume".into(),
            value: format!("{vr:.2}x avg"),
            lean,
        });
    }

    CompositeSignal {
        action: rule_output.action,
        confidence: rule_output.confidence,
        score: rule_output.score,
        rule_output,
        sources,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> AdvisorInput {
        AdvisorInput {
            symbol: "RELIANCE.NS".into(),
            last_price: 100.0,
            ..Default::default()
        }
    }

    #[test]
    fn verdict_is_delegated_and_never_recomputed() {
        let mut i = input();
        i.rsi = Some(25.0);
        i.forecast_change_pct = Some(5.0);
        let c = composite(&i);
        assert_eq!(c.action, Action::StrongBuy);
        assert_eq!(c.score, c.rule_output.score);
        assert_eq!(c.confidence, c.rule_output.confidence);
        assert_eq!(c.action, c.rule_output.action);
    }

    #[test]
    fn only_present_sources_appear_and_order_is_stable() {
        let mut i = input();
        i.rsi = Some(50.0);
        i.macd_signal = Some(0.1);
        i.forecast_change_pct = Some(1.0);
        let c = composite(&i);
        let names: Vec<&str> = c.sources.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["RSI", "MACD", "Forecast"]);
    }

    #[test]
    fn missing_inputs_produce_no_row() {
        let mut i = input();
        i.rsi = Some(50.0);
        let c = composite(&i);
        assert_eq!(c.sources.len(), 1, "{:?}", c.sources);
        assert!(!c.sources.iter().any(|s| s.name == "MACD"));
    }

    #[test]
    fn leans_point_the_same_way_as_the_verdict() {
        let mut i = input();
        i.rsi = Some(25.0);
        i.forecast_change_pct = Some(6.0);
        i.macd_signal = Some(0.5);
        i.price_vs_sma50 = Some(12.0);
        let c = composite(&i);
        assert_eq!(c.action, Action::StrongBuy);
        assert!(
            c.sources.iter().all(|s| s.lean > 0.0),
            "a STRONG BUY must not carry a bearish source: {:?}",
            c.sources
        );
    }

    #[test]
    fn forecast_lean_saturates_rather_than_exceeding_one() {
        let mut i = input();
        i.forecast_change_pct = Some(50.0);
        let c = composite(&i);
        assert!(c.sources.iter().all(|s| s.lean <= 1.0));
        let mut i = input();
        i.forecast_change_pct = Some(-50.0);
        let c = composite(&i);
        assert!(c.sources.iter().all(|s| s.lean >= -1.0));
    }

    #[test]
    fn agreement_is_zero_with_nothing_and_nothing_leaning() {
        assert_eq!(composite(&input()).agreement(), 0.0);
        let mut i = input();
        i.rsi = Some(50.0); // neutral, so it is present but not counted
        assert_eq!(composite(&i).agreement(), 0.0);
    }

    #[test]
    fn agreement_counts_the_dominant_direction() {
        // 3 bullish of 4 leaning sources -> 0.75.
        let mut i = input();
        i.rsi = Some(25.0); // +0.8
        i.forecast_change_pct = Some(5.0); // +1.0
        i.macd_signal = Some(0.5); // +0.5
        i.price_vs_sma50 = Some(-12.0); // -1.0
        i.volume_ratio = Some(0.2); // -0.2
        // Leaning sources: all five. Bullish 3, bearish 2.
        assert!((composite(&i).agreement() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn neutral_sources_do_not_dilute_agreement() {
        // RSI 50 and volume 1.0 are present but lean 0, so they must not
        // count against the direction.
        let mut i = input();
        i.rsi = Some(25.0); // +0.8
        i.volume_ratio = Some(1.0); // 0.0
        let c = composite(&i);
        assert!((c.agreement() - 1.0).abs() < 1e-9, "{c:?}");
    }

    #[test]
    fn non_finite_inputs_produce_no_nan_leans() {
        let mut i = input();
        i.rsi = Some(f64::NAN);
        i.forecast_change_pct = Some(f64::INFINITY);
        i.macd_signal = Some(f64::NAN);
        let c = composite(&i);
        assert!(c.sources.iter().all(|s| s.lean.is_finite()));
        assert!(c.sources.iter().all(|s| !s.value.contains("NaN")));
        assert!(c.sources.iter().all(|s| !s.value.contains("inf")));
    }

    #[test]
    fn composite_is_deterministic() {
        let mut i = input();
        i.rsi = Some(62.0);
        i.forecast_change_pct = Some(-1.5);
        i.macd_signal = Some(-0.2);
        assert_eq!(composite(&i), composite(&i));
    }
}
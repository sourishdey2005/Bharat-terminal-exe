// crates/bt-analytics/src/advisor.rs
// Author: Sourish Dey

//! Deterministic rule engine behind the Advisor tab.
//!
//! This does not forecast and it does not trade. It reads indicators the app
//! has already measured, scores them against fixed thresholds, and reports what
//! the data says. Given the same input it always returns the same output, which
//! is the point: a reader can audit the rules and disagree with a threshold,
//! but cannot be shown a different answer on a different day.
//!
//! The [`Action`] enum deliberately includes [`Action::Wait`] and
//! [`Action::Hold`] so that "the inputs do not point anywhere" is a first-class,
//! visible outcome rather than a fabricated confident call. Every branch that
//! moves the score also pushes a human-readable reason, so a score of +2 can
//! never appear in the UI without the two inputs that produced it.

use serde::{Deserialize, Serialize};

/// What the measured inputs point to.
///
/// `label` and `color` drive the Advisor badge; the palette is deliberately
/// close to the app's PROFIT/LOSS/AMBER so the badge reads consistently with
/// every other chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    StrongBuy,
    Buy,
    Hold,
    Reduce,
    Sell,
    StrongSell,
    /// Not enough measured inputs to say anything.
    Wait,
}

impl Action {
    /// Short uppercase label for the badge.
    pub fn label(&self) -> &'static str {
        match self {
            Action::StrongBuy => "STRONG BUY",
            Action::Buy => "BUY",
            Action::Hold => "HOLD",
            Action::Reduce => "REDUCE",
            Action::Sell => "SELL",
            Action::StrongSell => "STRONG SELL",
            Action::Wait => "WAIT",
        }
    }

    /// RGB the badge is drawn in.
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            Action::StrongBuy => (0, 200, 100),
            Action::Buy => (0, 230, 118),
            Action::Hold => (136, 146, 166),
            Action::Reduce => (255, 179, 0),
            Action::Sell => (255, 61, 113),
            Action::StrongSell => (200, 0, 60),
            Action::Wait => (120, 120, 120),
        }
    }
}

/// Everything the rule engine is allowed to look at.
///
/// Every field is optional on purpose: the Advisor tab renders before the
/// forecast and the indicators have all resolved, and a missing input must
/// degrade to "one fewer vote", never to a panic or a zero that reads as a
/// real reading.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdvisorInput {
    pub symbol: String,
    pub last_price: f64,
    pub rsi: Option<f64>,
    /// MACD minus its signal line. Positive is bullish.
    pub macd_signal: Option<f64>,
    /// Forecast move over the horizon, in percent.
    pub forecast_change_pct: Option<f64>,
    /// Verdict from the WatchSignal classifier, if one ran.
    pub watchsignal: Option<String>,
    /// Percent the last price sits above (+) or below (-) the 20-period SMA.
    pub price_vs_sma20: Option<f64>,
    /// Same, against the 50-period SMA.
    pub price_vs_sma50: Option<f64>,
    /// ATR expressed as a percent of price.
    pub atr_pct: Option<f64>,
    /// Latest volume divided by its 20-bar average.
    pub volume_ratio: Option<f64>,
}

/// What the rules concluded, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdvisorOutput {
    pub action: Action,
    /// 0..1. Scales with the magnitude of the score, not with data quality.
    pub confidence: f64,
    /// Each input that moved the score, with its reading.
    pub reasons: Vec<String>,
    /// Risk flags. These never move the score; they only qualify the result.
    pub warnings: Vec<String>,
    /// The raw net score, exposed so the UI can show why a threshold was hit.
    pub score: i32,
    /// How many inputs were actually present. `0` means [`Action::Wait`].
    pub inputs_seen: usize,
}

/// Score one indicator set and report the conclusion.
///
/// The thresholds are the numbers a reader is most likely to argue with, so
/// they are named constants rather than literals buried in branches: changing a
/// threshold should be a visible edit.
const RSI_OVERSOLD: f64 = 30.0;
const RSI_MILDLY_BULLISH: f64 = 45.0;
const RSI_MILDLY_BEARISH: f64 = 55.0;
const RSI_OVERBOUGHT: f64 = 70.0;

/// Apply the rule set to `input`.
///
/// Returns [`Action::Wait`] when nothing was measured — an empty score of 0 is
/// indistinguishable from a genuinely neutral read, and reporting "HOLD" for a
/// chart that has not loaded yet would be a confident answer to no question.
pub fn advise(input: &AdvisorInput) -> AdvisorOutput {
    let mut score: i32 = 0;
    let mut reasons: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut seen = 0usize;

    if let Some(rsi) = input.rsi.filter(|v| v.is_finite()) {
        seen += 1;
        if rsi < RSI_OVERSOLD {
            score += 2;
            reasons.push(format!("RSI {rsi:.1} — oversold"));
        } else if rsi < RSI_MILDLY_BULLISH {
            score += 1;
            reasons.push(format!("RSI {rsi:.1} — mildly bullish"));
        } else if rsi > RSI_OVERBOUGHT {
            score -= 2;
            reasons.push(format!("RSI {rsi:.1} — overbought"));
        } else if rsi > RSI_MILDLY_BEARISH {
            score -= 1;
            reasons.push(format!("RSI {rsi:.1} — mildly bearish"));
        } else {
            reasons.push(format!("RSI {rsi:.1} — neutral"));
        }
    }

    if let Some(fc) = input.forecast_change_pct.filter(|v| v.is_finite()) {
        seen += 1;
        if fc > 2.0 {
            score += 2;
            reasons.push(format!("Forecast {fc:+.2}% — bullish"));
        } else if fc > 0.5 {
            score += 1;
            reasons.push(format!("Forecast {fc:+.2}% — mildly bullish"));
        } else if fc < -2.0 {
            score -= 2;
            reasons.push(format!("Forecast {fc:+.2}% — bearish"));
        } else if fc < -0.5 {
            score -= 1;
            reasons.push(format!("Forecast {fc:+.2}% — mildly bearish"));
        } else {
            reasons.push(format!("Forecast {fc:+.2}% — flat"));
        }
    }

    if let Some(macd) = input.macd_signal.filter(|v| v.is_finite()) {
        seen += 1;
        if macd > 0.0 {
            score += 1;
            reasons.push(format!("MACD {macd:+.3} — above signal"));
        } else {
            score -= 1;
            reasons.push(format!("MACD {macd:+.3} — below signal"));
        }
    }

    if let Some(sig) = &input.watchsignal {
        seen += 1;
        let upper = sig.to_uppercase();
        if upper.contains("BUY") {
            score += 1;
            reasons.push(format!("WatchSignal: {sig}"));
        } else if upper.contains("SELL") {
            score -= 1;
            reasons.push(format!("WatchSignal: {sig}"));
        } else {
            reasons.push(format!("WatchSignal: {sig} (no direction)"));
        }
    }

    // Trend agreement: only the 50-period counts toward the score. SMA20 is
    // reported as context because it moves almost every day and would
    // otherwise swamp the slower, more meaningful vote.
    if let Some(p) = input.price_vs_sma50.filter(|v| v.is_finite()) {
        seen += 1;
        if p > 3.0 {
            score += 1;
            reasons.push(format!("Price {p:+.1}% above SMA50"));
        } else if p < -3.0 {
            score -= 1;
            reasons.push(format!("Price {p:+.1}% below SMA50"));
        } else {
            reasons.push(format!("Price {p:+.1}% vs SMA50 — near it"));
        }
    }

    if let Some(p) = input.price_vs_sma20.filter(|v| v.is_finite()) {
        seen += 1;
        reasons.push(format!("Price {p:+.1}% vs SMA20"));
    }

    // Warnings qualify the result without moving it: a high-ATR chart can be
    // strongly bullish and still be a bad thing to size into.
    if let Some(atr) = input.atr_pct.filter(|v| v.is_finite()) {
        seen += 1;
        if atr > 5.0 {
            warnings.push(format!("High volatility (ATR {atr:.1}% of price)"));
        }
    }

    if let Some(vr) = input.volume_ratio.filter(|v| v.is_finite()) {
        seen += 1;
        if vr < 0.5 {
            warnings.push(format!("Thin volume ({vr:.2}x the 20-bar average)"));
        }
    }

    let (action, confidence) = if seen == 0 {
        (Action::Wait, 0.0)
    } else {
        match score {
            4.. => (Action::StrongBuy, 0.9),
            2..=3 => (Action::Buy, 0.75),
            1 => (Action::Buy, 0.55),
            0 => (Action::Hold, 0.5),
            -1 => (Action::Reduce, 0.55),
            -3..=-2 => (Action::Sell, 0.75),
            _ => (Action::StrongSell, 0.9),
        }
    };

    if action == Action::Wait {
        warnings.push("No indicators measured yet".into());
    }

    AdvisorOutput {
        action,
        confidence,
        reasons,
        warnings,
        score,
        inputs_seen: seen,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: an input with only the fields a test cares about.
    fn input() -> AdvisorInput {
        AdvisorInput {
            symbol: "RELIANCE.NS".into(),
            last_price: 2950.75,
            ..Default::default()
        }
    }

    #[test]
    fn no_inputs_waits_rather_than_holding() {
        // An empty score of 0 must not read as a confident "HOLD": that would
        // be a confident answer to a chart that has not loaded.
        let out = advise(&input());
        assert_eq!(out.action, Action::Wait);
        assert_eq!(out.score, 0);
        assert_eq!(out.inputs_seen, 0);
        assert_eq!(out.confidence, 0.0);
        assert!(out.warnings.iter().any(|w| w.contains("No indicators")), "{:?}", out.warnings);
    }

    #[test]
    fn every_action_threshold_is_reachable() {
        // Each band is asserted through the score it corresponds to, so a
        // threshold edit that makes a band unreachable fails here.
        // +4: oversold RSI(+2) + bullish forecast(+2).
        let mut i = input();
        i.rsi = Some(25.0);
        i.forecast_change_pct = Some(5.0);
        assert_eq!(advise(&i).action, Action::StrongBuy);

        // +2: oversold RSI only.
        let mut i = input();
        i.rsi = Some(25.0);
        assert_eq!(advise(&i).action, Action::Buy);

        // +1: mildly bullish RSI.
        let mut i = input();
        i.rsi = Some(40.0);
        assert_eq!(advise(&i).action, Action::Buy);
        assert_eq!(advise(&i).confidence, 0.55);

        // 0: neutral RSI.
        let mut i = input();
        i.rsi = Some(50.0);
        assert_eq!(advise(&i).action, Action::Hold);
        assert_eq!(advise(&i).confidence, 0.5);

        // -1: mildly bearish RSI.
        let mut i = input();
        i.rsi = Some(60.0);
        assert_eq!(advise(&i).action, Action::Reduce);

        // -2: overbought RSI.
        let mut i = input();
        i.rsi = Some(75.0);
        assert_eq!(advise(&i).action, Action::Sell);
        assert_eq!(advise(&i).confidence, 0.75);

        // -4: overbought RSI + bearish forecast.
        let mut i = input();
        i.rsi = Some(75.0);
        i.forecast_change_pct = Some(-5.0);
        assert_eq!(advise(&i).action, Action::StrongSell);
    }

    #[test]
    fn boundary_values_take_the_documented_side() {
        // 30 is not oversold, 70 is not overbought: both are inside the
        // "mildly" bands, and the doc comment claims as much.
        let mut i = input();
        i.rsi = Some(30.0);
        assert_eq!(advise(&i).score, 1, "RSI 30 must not count as oversold");

        let mut i = input();
        i.rsi = Some(70.0);
        assert_eq!(advise(&i).score, -1, "RSI 70 must not count as overbought");

        // A forecast of exactly +2% is inside the mild band, not the strong one.
        let mut i = input();
        i.forecast_change_pct = Some(2.0);
        assert_eq!(advise(&i).score, 1);
    }

    #[test]
    fn every_scoring_input_records_a_reason() {
        // A score with no reason would show a number the reader cannot audit.
        let mut i = input();
        i.rsi = Some(25.0);
        i.forecast_change_pct = Some(5.0);
        i.macd_signal = Some(0.4);
        i.watchsignal = Some("BUY".into());
        i.price_vs_sma50 = Some(8.0);
        let out = advise(&i);
        assert_eq!(out.score, 7);
        // 5 scored inputs -> at least 5 reasons.
        assert!(out.reasons.len() >= 5, "{:?}", out.reasons);
    }

    #[test]
    fn sma20_is_context_only_and_never_moves_the_score() {
        let mut i = input();
        i.rsi = Some(50.0); // scores 0
        i.price_vs_sma20 = Some(99.0);
        let out = advise(&i);
        assert_eq!(out.score, 0, "SMA20 must not swing the score");
        assert!(out.reasons.iter().any(|r| r.contains("SMA20")), "{:?}", out.reasons);
    }

    #[test]
    fn warnings_qualify_without_moving_the_score() {
        let mut i = input();
        i.rsi = Some(25.0); // +2, would be BUY on its own
        i.atr_pct = Some(9.0);
        i.volume_ratio = Some(0.2);
        let out = advise(&i);
        assert_eq!(out.score, 2);
        assert_eq!(out.action, Action::Buy);
        assert!(out.warnings.iter().any(|w| w.contains("High volatility")), "{:?}", out.warnings);
        assert!(out.warnings.iter().any(|w| w.contains("Thin volume")), "{:?}", out.warnings);
    }

    #[test]
    fn non_finite_readings_are_ignored_not_scored() {
        // A NaN from an unfetched indicator must not poison the score: it is
        // dropped, so the remaining inputs still decide.
        let mut i = input();
        i.rsi = Some(25.0);
        i.forecast_change_pct = Some(f64::NAN);
        i.macd_signal = Some(f64::INFINITY);
        let out = advise(&i);
        assert_eq!(out.score, 2);
        assert_eq!(
            out.inputs_seen, 1,
            "only RSI survived the finite filter; forecast and MACD were dropped"
        );
    }

    #[test]
    fn identical_input_gives_an_identical_verdict() {
        let mut i = input();
        i.rsi = Some(58.0);
        i.forecast_change_pct = Some(-1.2);
        i.price_vs_sma50 = Some(4.5);
        let a = advise(&i);
        let b = advise(&i);
        assert_eq!(a.action, b.action);
        assert_eq!(a.score, b.score);
        assert_eq!(a.confidence, b.confidence);
    }

    #[test]
    fn every_action_has_a_distinct_label_and_colour() {
        let all = [
            Action::StrongBuy,
            Action::Buy,
            Action::Hold,
            Action::Reduce,
            Action::Sell,
            Action::StrongSell,
            Action::Wait,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.label(), b.label(), "{a:?} and {b:?} share a label");
                assert_ne!(a.color(), b.color(), "{a:?} and {b:?} share a colour");
            }
            assert!(!a.label().is_empty());
        }
    }

    #[test]
    fn watchsignal_matches_substrings_not_equality() {
        let mut i = input();
        i.watchsignal = Some("strong buy signal".into());
        assert_eq!(advise(&i).score, 1);
        i.watchsignal = Some("consider selling".into());
        assert_eq!(advise(&i).score, -1);
        i.watchsignal = Some("NEUTRAL".into());
        assert_eq!(advise(&i).score, 0);
    }
}
// crates/bt-analytics/src/risk_advisor.rs
// Author: Sourish Dey

//! Position sizing, stop placement, and the risk caps around them.
//!
//! Sizing here is deliberately conservative and *arithmetic*, not a Kelly
//! estimate. Full Kelly needs a real win rate and payoff distribution, neither
//! of which this app has measured reliably, and Kelly's answer to a wrong
//! win-rate estimate is a catastrophic position. So the size is driven by how
//! much capital the trader is willing to lose if the stop is hit, divided by
//! how far the stop sits, then scaled down by the advisor's confidence and
//! capped against the account.
//!
//! Nothing in this module is advice. It answers "if you took `risk_pct` of the
//! account as the loss you can tolerate and placed the stop at 2x ATR, here is
//! the size that loss implies" — the judgement stays with the caller.

/// Stop distance as a multiple of ATR.
pub const STOP_ATR_MULT: f64 = 2.0;

/// Target distance as a multiple of the stop distance (2:1 reward:risk).
pub const REWARD_RISK: f64 = 2.0;

/// Never let one position exceed this fraction of capital, however small the
/// stop looks. A very tight ATR must not be able to justify an outsized bet.
pub const MAX_CAPITAL_FRACTION: f64 = 0.25;

/// A concrete, inspectable trade plan.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskPlan {
    /// Position size in units of the instrument.
    pub position_size: f64,
    /// Where the stop sits.
    pub stop_loss: f64,
    /// Where the target sits.
    pub target: f64,
    /// Reward divided by risk. Constant by construction, reported so the UI
    /// can show it rather than hard-coding "2:1" in two places.
    pub risk_reward: f64,
    /// Capital at risk if the stop is hit.
    pub capital_at_risk: f64,
    /// Caps and assumptions that shaped the result.
    pub warnings: Vec<String>,
}

/// Build a sizing plan.
///
/// `atr` is the average true range in price units, not a percentage — mixing the
/// two silently produces a stop in the wrong place. `confidence` comes from
/// [`crate::advisor`] and scales the size down, but is clamped so a
/// low-confidence read cannot shrink a position to zero and report that as a
/// plan.
pub fn plan(
    capital: f64,
    risk_pct: f64,
    last_price: f64,
    atr: f64,
    confidence: f64,
) -> RiskPlan {
    let mut warnings = Vec::new();

    if !capital.is_finite() || capital <= 0.0 {
        warnings.push("No capital configured".into());
        return degenerate(capital, last_price, atr, warnings);
    }
    if !last_price.is_finite() || last_price <= 0.0 {
        warnings.push("No valid price".into());
        return degenerate(capital, last_price, atr, warnings);
    }
    if !atr.is_finite() || atr <= 0.0 {
        // Without a volatility estimate there is no defensible stop, and a
        // stop at zero distance implies infinite size. Refuse rather than guess.
        warnings.push("No ATR: cannot place a volatility-based stop".into());
        return degenerate(capital, last_price, atr, warnings);
    }

    // A stop at or below zero is not a stop, it is a guaranteed loss.
    let mut stop_distance = atr * STOP_ATR_MULT;
    let stop_loss = last_price - stop_distance;
    if stop_loss <= 0.0 {
        warnings.push(format!(
            "Stop at {stop_loss:.2} is not positive at {STOP_ATR_MULT}x ATR; falling back to a tighter stop"
        ));
        stop_distance = last_price * 0.5;
    }

    let risk_amount = (capital * risk_pct).max(0.0);
    if risk_pct <= 0.0 {
        warnings.push("Risk fraction is zero; position is zero".into());
    }
    if risk_pct > 0.02 {
        warnings.push(format!(
            "Risking {:.2}% per position is above the conventional 2% ceiling",
            risk_pct * 100.0
        ));
    }

    let mut size = risk_amount / stop_distance;

    // Confidence scales the size but cannot eliminate it, and cannot exceed 1.
    let scale = confidence.clamp(0.3, 1.0);
    size *= scale;
    if scale < 1.0 {
        warnings.push(format!(
            "Size scaled to {:.0}% of the risk budget by confidence {:.0}%",
            scale * 100.0,
            confidence * 100.0
        ));
    }

    let max_value = capital * MAX_CAPITAL_FRACTION;
    if size * last_price > max_value {
        size = max_value / last_price;
        warnings.push(format!(
            "Position capped at {:.0}% of capital",
            MAX_CAPITAL_FRACTION * 100.0
        ));
    }

    let stop_loss = last_price - stop_distance;
    let target = last_price + stop_distance * REWARD_RISK;

    if size <= 0.0 {
        warnings.push("Computed size is zero; no trade is implied".into());
    }

    RiskPlan {
        position_size: size,
        stop_loss,
        target,
        risk_reward: REWARD_RISK,
        capital_at_risk: size * stop_distance,
        warnings,
    }
}

/// A plan with no usable numbers, used for the refuse paths.
///
/// `stop_loss`/`target` are derived from `atr` only when `atr` is sane;
/// otherwise they collapse onto the price so the UI shows a flat plan instead
/// of NaN.
fn degenerate(
    _capital: f64,
    last_price: f64,
    atr: f64,
    mut warnings: Vec<String>,
) -> RiskPlan {
    let usable_price = if last_price.is_finite() && last_price > 0.0 {
        last_price
    } else {
        0.0
    };
    let stop_distance = if atr.is_finite() && atr > 0.0 {
        atr * STOP_ATR_MULT
    } else {
        0.0
    };
    warnings.push("No position sizing: inputs incomplete".into());
    RiskPlan {
        position_size: 0.0,
        stop_loss: usable_price - stop_distance,
        target: usable_price + stop_distance * REWARD_RISK,
        risk_reward: REWARD_RISK,
        capital_at_risk: 0.0,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

/// A 100k account risking 2% on a 1000-rupee stock with a 40-rupee ATR.
///
/// The ATR is deliberately ~4% of price: below roughly 3% of price the 25%
/// capital cap binds and the raw sizing formula stops being observable, so a
/// fixture any tighter would silently be testing the cap instead.
fn typical() -> (f64, f64, f64, f64, f64) {
    (100_000.0, 0.02, 1000.0, 40.0, 0.75)
}

    #[test]
    fn size_is_the_risk_budget_divided_by_the_stop_distance() {
        let (cap, risk, price, atr, conf) = typical();
        let p = plan(cap, risk, price, atr, conf);
        // Risk budget = 100000*0.02 = 2000. Stop distance = 2*40 = 80.
        // Size = 2000/80 = 25 units, scaled by confidence 0.75 -> 18.75.
        assert!((p.position_size - 18.75).abs() < 1e-9, "{p:?}");
        assert!((p.capital_at_risk - 1500.0).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn stop_and_target_sit_symmetrically_around_price() {
        let p = plan(100_000.0, 0.02, 1000.0, 40.0, 0.75);
        assert!((p.stop_loss - 920.0).abs() < 1e-9, "{p:?}");
        assert!((p.target - 1160.0).abs() < 1e-9, "{p:?}");
        assert!((p.risk_reward - 2.0).abs() < 1e-9);
        // Reward really is twice risk, measured from the price.
        let risk = 1000.0 - p.stop_loss;
        let reward = p.target - 1000.0;
        assert!((reward / risk - 2.0).abs() < 1e-9);
    }

    #[test]
    fn position_never_exceeds_the_capital_cap() {
        // A tiny ATR would otherwise justify an enormous position.
        let p = plan(100_000.0, 0.02, 1000.0, 0.01, 1.0);
        assert!(p.position_size * 1000.0 <= 100_000.0 * MAX_CAPITAL_FRACTION + 1e-9, "{p:?}");
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("capped")), "{:?}", p.warnings);
    }

    #[test]
    fn confidence_scales_size_but_never_to_zero() {
        let full = plan(100_000.0, 0.02, 1000.0, 40.0, 1.0);
        let half = plan(100_000.0, 0.02, 1000.0, 40.0, 0.5);
        assert!(half.position_size < full.position_size);
        // A 0 confidence is clamped to 0.3, so there is still a plan.
        let zero = plan(100_000.0, 0.02, 1000.0, 40.0, 0.0);
        assert!(zero.position_size > 0.0, "{zero:?}");
        assert!((zero.position_size / full.position_size - 0.3).abs() < 1e-9, "{zero:?}");
        // Above 1.0 confidence is clamped, never extrapolated.
        let over = plan(100_000.0, 0.02, 1000.0, 40.0, 5.0);
        assert!((over.position_size - full.position_size).abs() < 1e-9);
    }

    #[test]
    fn zero_risk_fraction_yields_no_position() {
        let p = plan(100_000.0, 0.0, 1000.0, 10.0, 0.75);
        assert_eq!(p.position_size, 0.0);
        assert_eq!(p.capital_at_risk, 0.0);
        assert!(p.warnings.iter().any(|w| w.contains("Risk fraction")), "{:?}", p.warnings);
    }

    #[test]
    fn missing_atr_refuses_rather_than_guessing_a_stop() {
        // The dangerous case: with ATR 0 the stop distance is 0 and the size
        // divides by zero. It must refuse instead.
        for atr in [0.0, f64::NAN, f64::INFINITY, -5.0] {
            let p = plan(100_000.0, 0.02, 1000.0, atr, 0.75);
            assert_eq!(p.position_size, 0.0, "atr={atr} produced a position");
            assert!(p.position_size.is_finite());
            assert!(
                p.warnings.iter().any(|w| w.contains("ATR")),
                "atr={atr} warnings: {:?}",
                p.warnings
            );
        }
    }

    #[test]
    fn bad_price_or_capital_yields_no_position_and_no_nan() {
        for (cap, price) in [
            (0.0, 1000.0),
            (-5.0, 1000.0),
            (f64::NAN, 1000.0),
            (100_000.0, 0.0),
            (100_000.0, -1.0),
            (100_000.0, f64::NAN),
        ] {
            let p = plan(cap, 0.02, price, 10.0, 0.75);
            assert_eq!(p.position_size, 0.0, "cap={cap} price={price}");
            assert!(p.stop_loss.is_finite(), "cap={cap} price={price}: {p:?}");
            assert!(p.target.is_finite(), "cap={cap} price={price}: {p:?}");
        }
    }

    #[test]
    fn a_stop_at_or_below_zero_is_reported_not_silently_used() {
        // 2x ATR below the price is negative here: a "stop" under zero is a
        // guaranteed loss, so the plan falls back and says so.
        let p = plan(100_000.0, 0.02, 10.0, 8.0, 0.75);
        assert!(
            p.warnings.iter().any(|w| w.contains("not positive")),
            "{:?}",
            p.warnings
        );
        assert!(p.stop_loss > 0.0, "{p:?}");
    }

    #[test]
    fn oversized_risk_fraction_is_flagged() {
        let p = plan(100_000.0, 0.10, 1000.0, 10.0, 0.75);
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("above the conventional 2%")), "{:?}", p.warnings);
    }

    #[test]
    fn plan_is_deterministic_for_identical_inputs() {
        let a = plan(100_000.0, 0.02, 1000.0, 10.0, 0.75);
        let b = plan(100_000.0, 0.02, 1000.0, 10.0, 0.75);
        assert_eq!(a, b);
    }

    #[test]
    fn capital_at_risk_never_exceeds_the_requested_budget() {
        // The whole point of the sizing formula: the loss if stopped out stays
        // within what the trader said they could lose.
        for atr in [1.0, 5.0, 10.0, 25.0, 50.0] {
            for conf in [0.3, 0.5, 0.75, 1.0] {
                let p = plan(100_000.0, 0.02, 1000.0, atr, conf);
                assert!(
                    p.capital_at_risk <= 2_000.0 + 1e-6,
                    "atr={atr} conf={conf} risked {:.2}",
                    p.capital_at_risk
                );
            }
        }
    }
}
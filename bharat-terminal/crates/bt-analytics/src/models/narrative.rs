// crates/bt-analytics/src/models/narrative.rs
// Author: Sourish Dey

//! Plain-language market commentary assembled from measured outputs.
//!
//! # Why this is not a language model
//!
//! `models/SmolLM2-135M-Instruct.Q4_K_M.gguf` (100 MB) ships alongside the
//! forecasters. It is deliberately **not** loaded here. On a 2 GB machine a 135M
//! GGUF plus an ONNX session is the whole budget spent, and a 135M instruct model
//! asked to describe a forecast invents numbers with more confidence than any it
//! was given. The commentary below is built from the same figures the chart draws,
//! so every number in a sentence is one the user can see on screen.
//!
//! If the GGUF is ever wired in, `MarketCommentary` is the shape it should fill,
//! and the fields below are the facts it is allowed to state.

use crate::signal::Signal;

/// A short written summary of a forecast.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketCommentary {
    /// One-line summary, suitable for a status bar.
    pub headline: String,
    /// The body: what was projected and against what.
    pub commentary: String,
    /// Coarse risk band derived from the classifier's confidence.
    pub risk_badge: String,
}

/// Builds [`MarketCommentary`] from measured values.
pub struct NarrativeEngine;

/// Confidence above which risk is called LOW.
pub const RISK_LOW_ABOVE: f32 = 0.70;
/// Confidence above which risk is called MEDIUM.
pub const RISK_MEDIUM_ABOVE: f32 = 0.50;

impl NarrativeEngine {
    /// Describe a point forecast.
    ///
    /// `current` is the last observed close and `target` the price being quoted,
    /// which is the final projected step for a projection and the last close for
    /// a directional call. `confidence` comes from the classifier, so the risk
    /// band and the headline agree with the signal badge.
    #[allow(clippy::too_many_arguments)]
    pub fn describe(
        symbol: &str,
        current: f64,
        target: f64,
        signal: Signal,
        confidence: f32,
        engine: &str,
    ) -> MarketCommentary {
        // Both operands must be usable: guarding only the anchor lets an infinite
        // target print "+inf%".
        let delta_pct = if current.is_finite() && target.is_finite() && current.abs() > 1e-12 {
            (target / current - 1.0) * 100.0
        } else {
            0.0
        };
        let risk = Self::risk_band(confidence);
        let headline = format!("{symbol} — {} ({:.0}%)", signal.label(), confidence * 100.0);
        // A non-finite price must never reach the page. Formatting `NaN` into a
        // sentence the user reads as a quote is worse than saying "n/a".
        let current_txt = fmt_price(current);
        let target_txt = fmt_price(target);
        let commentary = format!(
            "{engine} projects {delta_pct:+.2}% to a terminal level of {target_txt} \
             against a last close of {current_txt}. Confidence {:.0}% reads as {} risk.",
            confidence * 100.0,
            risk.to_lowercase()
        );
        MarketCommentary {
            headline,
            commentary,
            risk_badge: risk.to_string(),
        }
    }

    /// Describe a quantile corridor rather than a single target.
    ///
    /// A cone's honest summary is its width, not its midpoint: quoting only the
    /// median would hide exactly the information the cone exists to show.
    pub fn describe_cone(
        symbol: &str,
        current: f64,
        p10: f64,
        p50: f64,
        p90: f64,
        engine: &str,
    ) -> MarketCommentary {
        let width_pct = if current.is_finite() && current.abs() > 1e-12 {
            (p90 - p10).abs() / current.abs() * 100.0
        } else {
            0.0
        };
        let median_pct = if current.is_finite() && current.abs() > 1e-12 {
            (p50 / current - 1.0) * 100.0
        } else {
            0.0
        };
        // Width, not confidence, drives the band here: a narrow cone is low risk
        // whatever the classifier says.
        let risk = if width_pct < 3.0 {
            "LOW"
        } else if width_pct < 8.0 {
            "MEDIUM"
        } else {
            "HIGH"
        };
        MarketCommentary {
            headline: format!(
                "{symbol} — {engine} cone {median_pct:+.2}% median ({width_pct:.1}% wide)"
            ),
            commentary: format!(
                "80% of outcomes fall between {} and {}, a {width_pct:.1}% \
                 corridor around a {} median, from a last close of {}.",
                fmt_price(p10),
                fmt_price(p90),
                fmt_price(p50),
                fmt_price(current),
            ),
            risk_badge: risk.to_string(),
        }
    }

    /// Confidence to a coarse risk band.
    pub fn risk_band(confidence: f32) -> &'static str {
        if confidence > RISK_LOW_ABOVE {
            "LOW"
        } else if confidence > RISK_MEDIUM_ABOVE {
            "MEDIUM"
        } else {
            "HIGH"
        }
    }
}

/// Render a price for prose, refusing to print a non-finite value.
fn fmt_price(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.2}")
    } else {
        "n/a".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_band_boundaries() {
        assert_eq!(NarrativeEngine::risk_band(0.95), "LOW");
        assert_eq!(NarrativeEngine::risk_band(0.70), "MEDIUM");
        assert_eq!(NarrativeEngine::risk_band(0.50), "HIGH");
        assert_eq!(NarrativeEngine::risk_band(0.0), "HIGH");
        // A confidence above 1 cannot make a "low risk" claim.
        assert_eq!(NarrativeEngine::risk_band(2.0), "LOW");
    }

    #[test]
    fn every_figure_in_the_commentary_comes_from_the_input() {
        let c = NarrativeEngine::describe("TEST.NS", 100.0, 110.0, Signal::Buy, 0.8, "Chronos");
        // 110 / 100 - 1 = +10%.
        assert!(c.commentary.contains("+10.00"), "{}", c.commentary);
        assert!(c.commentary.contains("110.00"));
        assert!(c.commentary.contains("100.00"));
        assert!(c.commentary.contains("Chronos"));
        assert!(c.headline.contains("BUY"));
        assert_eq!(c.risk_badge, "LOW");
    }

    #[test]
    fn a_falling_projection_is_reported_as_negative() {
        let c = NarrativeEngine::describe("X", 100.0, 90.0, Signal::Sell, 0.6, "ARIMA");
        assert!(c.commentary.contains("-10.00"), "{}", c.commentary);
        assert_eq!(c.risk_badge, "MEDIUM");
    }

    #[test]
    fn a_zero_price_does_not_divide_by_zero() {
        let c = NarrativeEngine::describe("X", 0.0, 10.0, Signal::Hold, 0.5, "m");
        assert!(c.commentary.contains("+0.00%"), "{}", c.commentary);
        assert!(!c.commentary.contains("NaN"));
        assert!(!c.commentary.contains("inf"));
    }

    #[test]
    fn cone_commentary_quotes_the_width_not_just_the_median() {
        let c = NarrativeEngine::describe_cone("ABC", 100.0, 95.0, 101.0, 108.0, "Chronos");
        // Width = 13 / 100 = 13% -> HIGH.
        assert!(c.commentary.contains("13.0%"), "{}", c.commentary);
        assert!(c.commentary.contains("95.00"));
        assert!(c.commentary.contains("108.00"));
        assert_eq!(c.risk_badge, "HIGH");
        assert!(c.headline.contains("Chronos"));
    }

    #[test]
    fn a_narrow_cone_is_low_risk() {
        // 2.5% wide, comfortably inside the <3% band.
        let c = NarrativeEngine::describe_cone("ABC", 100.0, 99.0, 100.0, 101.5, "Chronos");
        assert_eq!(c.risk_badge, "LOW");
        assert!(c.commentary.contains("2.5%"), "{}", c.commentary);
    }

    /// A price that is not a number must not be printed as one.
    #[test]
    fn no_nan_leaks_into_any_commentary() {
        for (cur, tgt) in [
            (0.0, 0.0),
            (100.0, 0.0),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            let c = NarrativeEngine::describe("N", cur, tgt, Signal::Hold, 0.5, "m");
            assert!(
                !c.commentary.contains("NaN"),
                "cur={cur} -> {}",
                c.commentary
            );
            assert!(
                !c.commentary.contains("inf"),
                "cur={cur} -> {}",
                c.commentary
            );
        }
        let c = NarrativeEngine::describe_cone("N", f64::NAN, f64::NAN, 1.0, f64::NAN, "m");
        assert!(!c.commentary.contains("NaN"), "{}", c.commentary);
        assert!(c.commentary.contains("n/a"), "{}", c.commentary);
    }
}

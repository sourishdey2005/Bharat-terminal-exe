// crates/bt-app/src/panel_advisor.rs
// Author: Sourish Dey

//! The Advisor tab: what the deterministic rules concluded, and why.
//!
//! The panel is a readout, not a widget. Every number on it comes from
//! [`bt_analytics::advisor`] and [`bt_analytics::signals`], computed on the
//! candles the app already has. Nothing here forecasts, nothing here trades, and
//! the action badge is labelled with the score that produced it so the verdict is
//! auditable rather than oracular.
//!
//! Layout is top-down and deliberately plain: verdict, evidence, risk plan,
//! disclaimer. The disclaimer is not decoration — the rules are mechanical, and a
//! mechanical verdict in a professional-looking panel is exactly the thing a
//! reader might mistake for a recommendation.

use bt_analytics::advisor::{Action, AdvisorInput, AdvisorOutput};
use bt_analytics::risk_advisor::{RiskPlan, plan as risk_plan};
use bt_analytics::signals::{CompositeSignal, composite};
use bt_core::Candle;
use eframe::egui;

/// Fraction of trading capital the risk plan treats as per-trade risk.
pub const RISK_FRACTION: f64 = 0.02;

/// Capital used for the sizing example when the portfolio is empty.
///
/// A number has to come from somewhere, and silently sizing against a zero
/// account would render a plan of zero shares with no explanation. This is
/// labelled in the UI as an example rather than presented as the user's account.
pub const EXAMPLE_CAPITAL: f64 = 100_000.0;

/// Everything the panel draws from.
pub struct AdvisorView<'a> {
    pub symbol: &'a str,
    pub candles: &'a [Candle],
    /// Forecast change over the horizon, in percent, if one has been computed.
    pub forecast_change_pct: Option<f64>,
    /// Verdict from the WatchSignal classifier, if one ran.
    pub watchsignal: Option<String>,
    /// Account value used for the sizing example.
    pub capital: f64,
}

/// Build the rule input for the currently viewed symbol.
///
/// Delegates the indicator maths to [`bt_analytics::multi_scanner::advisor_input_for`]
/// so the Advisor tab and the watchlist sweep cannot drift apart: both are
/// reading the same indicators through the same code.
pub fn build_input(view: &AdvisorView<'_>) -> AdvisorInput {
    let mut input =
        bt_analytics::multi_scanner::advisor_input_for(view.symbol, view.candles);
    input.forecast_change_pct = view.forecast_change_pct;
    input.watchsignal = view.watchsignal.clone();
    input
}

/// The full advisor result for a view.
pub struct AdvisorResult {
    pub signal: CompositeSignal,
    pub plan: RiskPlan,
}

impl AdvisorResult {
    /// Verdict and the evidence behind it.
    pub fn output(&self) -> &AdvisorOutput {
        &self.signal.rule_output
    }

    /// Action the rules concluded.
    pub fn action(&self) -> Action {
        self.signal.action
    }
}

/// Compute the advisor result for a view.
pub fn evaluate(view: &AdvisorView<'_>) -> AdvisorResult {
    let input = build_input(view);
    let signal = composite(&input);

    // ATR drives the stop, so read it from the same indicators the rules used.
    let atr = last_atr(view.candles);
    let plan = risk_plan(
        view.capital,
        RISK_FRACTION,
        input.last_price,
        atr,
        signal.confidence,
    );
    AdvisorResult { signal, plan }
}

/// Latest ATR from the viewed candles.
fn last_atr(candles: &[Candle]) -> f64 {
    if candles.len() < 15 {
        return 0.0;
    }
    let series = bt_core::OhlcvSeries::new("advisor", candles.to_vec());
    bt_analytics::indicators::atr(&series, 14)
        .iter()
        .rev()
        .copied()
        .find(|v| v.is_finite())
        .unwrap_or(0.0)
}

/// Colour for the action badge.
fn action_color(action: Action) -> egui::Color32 {
    let (r, g, b) = action.color();
    egui::Color32::from_rgb(r, g, b)
}

/// Draw the Advisor panel. Returns true when the user pressed "Explain".
pub fn draw(ui: &mut egui::Ui, view: &AdvisorView<'_>) -> bool {
    let result = evaluate(view);
    let out = result.output();

    ui.label(
        egui::RichText::new("Advisor — deterministic rules over the measured data")
            .strong(),
    );
ui.label(
            egui::RichText::new(
                "Not financial advice. The rules are fixed and auditable; they explain data, they do not predict.",
            )
            .small()
            .color(egui::Color32::GRAY),
        );
    ui.separator();

    // Verdict.
    ui.horizontal(|ui| {
        let color = action_color(result.action());
        ui.label(
            egui::RichText::new(format!("● {}", result.action().label()))
                .color(color)
                .size(20.0),
        );
        ui.label(
            egui::RichText::new(format!("confidence {:.0}%", out.confidence * 100.0))
                .color(egui::Color32::GRAY),
        );
    });
    ui.label(
        egui::RichText::new(format!(
            "score {} from {} measured input{}",
            out.score,
            out.inputs_seen,
            if out.inputs_seen == 1 { "" } else { "s" }
        ))
        .small()
        .color(egui::Color32::GRAY),
    );

    // A WAIT verdict is the absence of data, not a neutral reading; say so
    // rather than leaving the badge to imply the chart is merely uninteresting.
    if result.action() == Action::Wait {
        ui.label(
            egui::RichText::new("Not enough history yet to run the rules.")
                .color(egui::Color32::from_rgb(255, 176, 0)),
        );
    }

    ui.add_space(6.0);
    ui.label(egui::RichText::new("Why").strong());
    if out.reasons.is_empty() {
        ui.label(
            egui::RichText::new("No indicators resolved yet.")
                .small()
                .color(egui::Color32::GRAY),
        );
    } else {
        egui::ScrollArea::vertical()
            .max_height(140.0)
            .show(ui, |ui| {
                for r in &out.reasons {
                    ui.label(format!("• {r}"));
                }
            });
    }

    if !out.warnings.is_empty() {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Risk flags").strong().color(egui::Color32::from_rgb(255, 176, 0)),
        );
        for w in &out.warnings {
            ui.label(egui::RichText::new(format!("⚠ {w}")).color(egui::Color32::from_rgb(255, 176, 0)));
        }
    }

    // Risk plan.
    ui.add_space(6.0);
    ui.separator();
    ui.label(egui::RichText::new("Risk plan (worked example)").strong());
    let plan = &result.plan;
    egui::Grid::new("advisor_risk_plan")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            ui.label("Position size");
            ui.label(format!("{:.2} units", plan.position_size));
            ui.end_row();
            ui.label("Stop loss");
            ui.label(format!("{:.2}", plan.stop_loss));
            ui.end_row();
            ui.label("Target");
            ui.label(format!("{:.2}", plan.target));
            ui.end_row();
            ui.label("Reward : risk");
            ui.label(format!("{:.1} : 1", plan.risk_reward));
            ui.end_row();
            ui.label("Capital at risk");
            ui.label(format!("{:.0}", plan.capital_at_risk));
            ui.end_row();
        });

    for w in &plan.warnings {
        ui.label(
            egui::RichText::new(format!("⚠ {w}"))
                .small()
                .color(egui::Color32::from_rgb(255, 176, 0)),
        );
    }

    ui.separator();
    ui.label(
        egui::RichText::new("Made by Sourish Dey")
            .size(10.0)
            .color(egui::Color32::from_rgb(255, 176, 0)),
    );

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rising series with alternating days, as in the multi_scanner fixtures.
    fn uptrend(n: usize) -> Vec<Candle> {
        let mut close = 100.0;
        (0..n)
            .map(|i| {
                let open = close;
                close += if i % 2 == 0 { -0.8 } else { 0.9 };
                Candle::new(
                    i as f64,
                    open,
                    close.max(open) + 0.3,
                    close.min(open) - 0.3,
                    close,
                    1000.0,
                )
            })
            .collect()
    }

    fn view<'a>(symbol: &'a str, candles: &'a [Candle]) -> AdvisorView<'a> {
        AdvisorView {
            symbol,
            candles,
            forecast_change_pct: None,
            watchsignal: None,
            capital: EXAMPLE_CAPITAL,
        }
    }

    #[test]
    fn a_short_history_does_not_panic_and_reports_wait() {
        let cs = uptrend(4);
        let r = evaluate(&view("X", &cs));
        assert_eq!(r.action(), Action::Wait, "{:?}", r.output());
        assert_eq!(r.output().inputs_seen, 0);
    }

    #[test]
    fn an_empty_series_does_not_panic() {
        let cs: Vec<Candle> = Vec::new();
        let r = evaluate(&view("X", &cs));
        assert_eq!(r.action(), Action::Wait);
    }

    #[test]
    fn a_real_history_produces_a_verdict_and_a_plan() {
        let cs = uptrend(120);
        let r = evaluate(&view("UP", &cs));
        assert_ne!(r.action(), Action::Wait, "{:?}", r.output());
        assert!(r.plan.position_size > 0.0, "{:?}", r.plan);
        assert!(r.plan.stop_loss < r.plan.target, "{:?}", r.plan);
    }

    #[test]
    fn a_forecast_supplied_by_the_app_reaches_the_rules() {
        let cs = uptrend(120);
        let mut v = view("X", &cs);
        let neutral = evaluate(&v).signal.score;
        v.forecast_change_pct = Some(6.0);
        let bullish = evaluate(&v).signal.score;
        assert!(bullish > neutral, "forecast must move the score: {bullish} vs {neutral}");
    }

    #[test]
    fn the_evidence_table_and_the_verdict_agree() {
        let cs = uptrend(120);
        let r = evaluate(&view("X", &cs));
        assert_eq!(
            r.signal.action,
            r.signal.rule_output.action,
            "the composite must not re-score"
        );
        assert_eq!(r.signal.sources.len(), r.output().inputs_seen.min(r.signal.sources.len()));
    }

    #[test]
    fn zero_capital_produces_no_position_rather_than_a_division_by_zero() {
        let cs = uptrend(120);
        let mut v = view("X", &cs);
        v.capital = 0.0;
        let r = evaluate(&v);
        assert_eq!(r.plan.position_size, 0.0);
        assert!(r.plan.stop_loss.is_finite());
        assert!(r.plan.target.is_finite());
    }

    #[test]
    fn evaluation_is_deterministic() {
        let cs = uptrend(120);
        let a = evaluate(&view("X", &cs));
        let b = evaluate(&view("X", &cs));
        assert_eq!(a.action(), b.action());
        assert_eq!(a.plan, b.plan);
    }
}
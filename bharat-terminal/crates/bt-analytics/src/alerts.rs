// crates/bt-analytics/src/alerts.rs
// Author: Sourish Dey

//! Alert rule evaluation (pure logic, no I/O).
//!
//! The [`AlertRule`]s are persisted by the app to `alerts.json`; this module
//! only decides whether a rule fires given a [`MarketSnapshot`]. Keeping
//! evaluation pure means the whole engine is unit-testable without market data,
//! models or a clock beyond the timestamps the caller supplies.
//!
//! Cooldowns live on the rule (`last_fired`) so a rule that stays true across
//! refresh cycles toasts once instead of every 30 seconds.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// What a rule watches. Thresholds are in the units the trigger reads:
/// prices in quote currency, moves in percent, RSI on 0–100.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertKind {
    /// Last close crossed above `level`.
    PriceAbove { level: f64 },
    /// Last close crossed below `level`.
    PriceBelow { level: f64 },
    /// Absolute move of at least `pct` percent over the last `bars` bars.
    PctMove { pct: f64, bars: usize },
    /// Latest volume exceeds `multiple` × the mean of the prior `window` bars.
    VolumeSpike { multiple: f64, window: usize },
    /// RSI(14) above `level` (overbought).
    RsiAbove { level: f64 },
    /// RSI(14) below `level` (oversold).
    RsiBelow { level: f64 },
    /// MACD line crossed above its signal line on the latest bar.
    MacdBullCross,
    /// MACD line crossed below its signal line on the latest bar.
    MacdBearCross,
    /// Close broke above the upper Bollinger band.
    BollingerBreakUpper,
    /// Close broke below the lower Bollinger band.
    BollingerBreakLower,
    /// A forecaster projects at least `pct` percent over `horizon` bars
    /// (signed: negative watches for drops).
    ForecastMove { pct: f64, horizon: usize },
    /// The classifier currently reads BUY (or SELL when `buy` is false).
    SignalIs { buy: bool },
}

impl AlertKind {
    /// Short human label for the rule list, e.g. `Price > 1500.00`.
    pub fn label(&self) -> String {
        match self {
            AlertKind::PriceAbove { level } => format!("Price > {level:.2}"),
            AlertKind::PriceBelow { level } => format!("Price < {level:.2}"),
            AlertKind::PctMove { pct, bars } => format!("Move ≥ {pct:.1}% / {bars} bars"),
            AlertKind::VolumeSpike { multiple, window } => {
                format!("Volume > {multiple:.1}x {window}-bar avg")
            }
            AlertKind::RsiAbove { level } => format!("RSI > {level:.0}"),
            AlertKind::RsiBelow { level } => format!("RSI < {level:.0}"),
            AlertKind::MacdBullCross => "MACD bull cross".to_string(),
            AlertKind::MacdBearCross => "MACD bear cross".to_string(),
            AlertKind::BollingerBreakUpper => "Breaks upper BB".to_string(),
            AlertKind::BollingerBreakLower => "Breaks lower BB".to_string(),
            AlertKind::ForecastMove { pct, horizon } => {
                format!("Forecast {pct:+.1}% / {horizon} bars")
            }
            AlertKind::SignalIs { buy } => {
                if *buy {
                    "Signal BUY".to_string()
                } else {
                    "Signal SELL".to_string()
                }
            }
        }
    }
}

/// One persisted rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: u64,
    pub symbol: String,
    pub kind: AlertKind,
    pub enabled: bool,
    /// Minimum seconds between firings. Zero means every evaluation may fire.
    #[serde(default)]
    pub cooldown_secs: u64,
    /// Last firing time; runtime state persisted so a restart does not re-toast.
    pub last_fired: Option<DateTime<Utc>>,
}

impl AlertRule {
    pub fn new(id: u64, symbol: &str, kind: AlertKind) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            kind,
            enabled: true,
            cooldown_secs: 3600,
            last_fired: None,
        }
    }

    /// Whether the cooldown has elapsed as of `now`.
    pub fn cooldown_elapsed(&self, now: DateTime<Utc>) -> bool {
        match self.last_fired {
            None => true,
            Some(t) => now
                .signed_duration_since(t)
                .num_seconds()
                .max(0) as u64
                >= self.cooldown_secs,
        }
    }
}

/// Everything a rule may need, computed once per evaluation pass by the caller.
#[derive(Debug, Clone, Default)]
pub struct MarketSnapshot {
    pub closes: Vec<f64>,
    pub volumes: Vec<f64>,
    pub rsi14: Option<f64>,
    /// (macd line, signal line) on the latest bar.
    pub macd: Option<(f64, f64)>,
    /// (macd line, signal line) on the previous bar, for cross detection.
    pub prev_macd: Option<(f64, f64)>,
    /// (lower, middle, upper) Bollinger bands on the latest bar.
    pub bollinger: Option<(f64, f64, f64)>,
    /// Projected percent move over the rule's horizon, if a forecaster ran.
    pub forecast_pct: Option<f64>,
    /// Latest classifier verdict, `"BUY"` / `"SELL"` / `"HOLD"`.
    pub signal: Option<String>,
}

/// A fired rule, ready to toast and persist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertEvent {
    pub rule_id: u64,
    pub symbol: String,
    pub message: String,
    pub at: DateTime<Utc>,
}

/// Evaluate one rule. Returns the event message when the condition holds *and*
/// the cooldown has elapsed; `None` otherwise (condition false, rule disabled,
/// or cooling down). `last_fired` is updated on fire so callers just persist.
pub fn evaluate(rule: &mut AlertRule, snap: &MarketSnapshot, now: DateTime<Utc>) -> Option<AlertEvent> {
    if !rule.enabled || !rule.cooldown_elapsed(now) {
        return None;
    }
    let message = condition_message(&rule.kind, &rule.symbol, snap)?;
    rule.last_fired = Some(now);
    Some(AlertEvent {
        rule_id: rule.id,
        symbol: rule.symbol.clone(),
        message,
        at: now,
    })
}

/// Pure condition check, exported so the UI can preview "would fire now".
pub fn condition_message(kind: &AlertKind, symbol: &str, snap: &MarketSnapshot) -> Option<String> {
    let last = *snap.closes.last()?;
    match kind {
        AlertKind::PriceAbove { level } if last > *level => {
            Some(format!("{symbol} {last:.2} crossed above {level:.2}"))
        }
        AlertKind::PriceBelow { level } if last < *level => {
            Some(format!("{symbol} {last:.2} crossed below {level:.2}"))
        }
        AlertKind::PctMove { pct, bars } => {
            let n = (*bars).max(1).min(snap.closes.len().saturating_sub(1));
            if n == 0 {
                return None;
            }
            let base = snap.closes[snap.closes.len() - 1 - n];
            if base.abs() < 1e-12 {
                return None;
            }
            let move_pct = (last / base - 1.0) * 100.0;
            if move_pct.abs() >= *pct {
                Some(format!("{symbol} moved {move_pct:+.2}% over {n} bars"))
            } else {
                None
            }
        }
        AlertKind::VolumeSpike { multiple, window } => {
            let w = (*window).max(1);
            if snap.volumes.len() <= w {
                return None;
            }
            let base: f64 =
                snap.volumes[snap.volumes.len() - 1 - w..snap.volumes.len() - 1].iter().sum::<f64>()
                    / w as f64;
            let last_v = *snap.volumes.last()?;
            if base > 0.0 && last_v > *multiple * base {
                Some(format!(
                    "{symbol} volume {last_v:.0} is {:.1}x the {w}-bar average",
                    last_v / base
                ))
            } else {
                None
            }
        }
        AlertKind::RsiAbove { level } => snap
            .rsi14
            .filter(|r| *r > *level)
            .map(|r| format!("{symbol} RSI {r:.1} above {level:.0}")),
        AlertKind::RsiBelow { level } => snap
            .rsi14
            .filter(|r| *r < *level)
            .map(|r| format!("{symbol} RSI {r:.1} below {level:.0}")),
        AlertKind::MacdBullCross => match (snap.prev_macd, snap.macd) {
            (Some((pl, ps)), Some((l, s))) if pl <= ps && l > s => {
                Some(format!("{symbol} MACD crossed bullish"))
            }
            _ => None,
        },
        AlertKind::MacdBearCross => match (snap.prev_macd, snap.macd) {
            (Some((pl, ps)), Some((l, s))) if pl >= ps && l < s => {
                Some(format!("{symbol} MACD crossed bearish"))
            }
            _ => None,
        },
        AlertKind::BollingerBreakUpper => match snap.bollinger {
            Some((_, _, upper)) if last > upper => {
                Some(format!("{symbol} {last:.2} broke above upper BB {upper:.2}"))
            }
            _ => None,
        },
        AlertKind::BollingerBreakLower => match snap.bollinger {
            Some((lower, _, _)) if last < lower => {
                Some(format!("{symbol} {last:.2} broke below lower BB {lower:.2}"))
            }
            _ => None,
        },
        AlertKind::ForecastMove { pct, .. } => snap.forecast_pct.and_then(|f| {
            if (*pct >= 0.0 && f >= *pct) || (*pct < 0.0 && f <= *pct) {
                Some(format!("{symbol} forecast {f:+.2}% meets {pct:+.1}%"))
            } else {
                None
            }
        }),
        AlertKind::SignalIs { buy } => snap.signal.as_deref().and_then(|s| {
            let want = if *buy { "BUY" } else { "SELL" };
            if s.eq_ignore_ascii_case(want) {
                Some(format!("{symbol} signal reads {want}"))
            } else {
                None
            }
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 30, 12, 0, 0).unwrap()
    }

    fn snap() -> MarketSnapshot {
        MarketSnapshot {
            closes: (0..60).map(|i| 100.0 + i as f64).collect(),
            volumes: vec![1000.0; 60],
            rsi14: Some(72.0),
            macd: Some((1.2, 1.0)),
            prev_macd: Some((0.9, 1.0)),
            bollinger: Some((150.0, 155.0, 160.0)),
            forecast_pct: Some(3.5),
            signal: Some("BUY".to_string()),
        }
    }

    #[test]
    fn price_triggers_fire_with_values_in_message() {
        let mut r = AlertRule::new(1, "RELIANCE.NS", AlertKind::PriceAbove { level: 150.0 });
        let ev = evaluate(&mut r, &snap(), now()).unwrap();
        assert!(ev.message.contains("159.00"), "{}", ev.message);
        // Cooling down: same condition must not fire twice.
        assert!(evaluate(&mut r, &snap(), now()).is_none());
    }

    #[test]
    fn disabled_rules_never_fire() {
        let mut r = AlertRule::new(2, "X", AlertKind::PriceBelow { level: 1000.0 });
        r.enabled = false;
        assert!(evaluate(&mut r, &snap(), now()).is_none());
        assert!(r.last_fired.is_none());
    }

    #[test]
    fn pct_move_uses_the_requested_window() {
        let s = snap();
        let m = condition_message(&AlertKind::PctMove { pct: 5.0, bars: 10 }, "X", &s).unwrap();
        assert!(m.contains("+"), "{m}");
        assert!(condition_message(&AlertKind::PctMove { pct: 90.0, bars: 10 }, "X", &s).is_none());
    }

    #[test]
    fn volume_spike_needs_history() {
        let mut s = snap();
        s.volumes = vec![1000.0; 5];
        s.volumes.push(5000.0);
        assert!(condition_message(
            &AlertKind::VolumeSpike { multiple: 3.0, window: 5 },
            "X",
            &s
        )
        .is_some());
        s.volumes = vec![1000.0; 3];
        assert!(condition_message(
            &AlertKind::VolumeSpike { multiple: 3.0, window: 5 },
            "X",
            &s
        )
        .is_none());
    }

    #[test]
    fn crosses_and_breaks() {
        let s = snap();
        assert!(condition_message(&AlertKind::MacdBullCross, "X", &s).is_some());
        assert!(condition_message(&AlertKind::MacdBearCross, "X", &s).is_none());
        assert!(condition_message(&AlertKind::RsiAbove { level: 70.0 }, "X", &s).is_some());
        assert!(condition_message(&AlertKind::RsiBelow { level: 30.0 }, "X", &s).is_none());
        // last close 159 < upper 160: no break; push it over.
        let mut s2 = s.clone();
        s2.closes.push(161.0);
        assert!(condition_message(&AlertKind::BollingerBreakUpper, "X", &s2).is_some());
    }

    #[test]
    fn forecast_and_signal_follow_sign_and_case() {
        let s = snap();
        assert!(condition_message(
            &AlertKind::ForecastMove { pct: 2.0, horizon: 20 },
            "X",
            &s
        )
        .is_some());
        // Watching for a -5% drop must not fire on a +3.5% projection.
        assert!(condition_message(
            &AlertKind::ForecastMove { pct: -5.0, horizon: 20 },
            "X",
            &s
        )
        .is_none());
        assert!(condition_message(&AlertKind::SignalIs { buy: true }, "X", &s).is_some());
        assert!(condition_message(&AlertKind::SignalIs { buy: false }, "X", &s).is_none());
    }

    #[test]
    fn empty_history_fires_nothing() {
        let mut r = AlertRule::new(9, "X", AlertKind::PriceAbove { level: 1.0 });
        let empty = MarketSnapshot::default();
        assert!(evaluate(&mut r, &empty, now()).is_none());
    }

    #[test]
    fn rules_round_trip_through_json() {
        let r = AlertRule::new(7, "TCS.NS", AlertKind::RsiBelow { level: 30.0 });
        let json = serde_json::to_string(&r).unwrap();
        let back: AlertRule = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn labels_are_short_and_greppable() {
        assert_eq!(AlertKind::PriceAbove { level: 1500.0 }.label(), "Price > 1500.00");
        assert_eq!(AlertKind::SignalIs { buy: false }.label(), "Signal SELL");
    }
}

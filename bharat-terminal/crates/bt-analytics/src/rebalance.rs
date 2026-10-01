// crates/bt-analytics/src/rebalance.rs
// Author: Sourish Dey

//! Portfolio rebalancing: what trades would move a book back to its targets.
//!
//! This produces *trade suggestions*, not orders, and it never decides the
//! targets themselves — the caller supplies them. The arithmetic is the whole
//! job: convert current weights and target weights into the buys and sells that
//! close the gap.
//!
//! Two rules keep the output honest:
//!
//! - Weights are normalized before anything else. A caller whose targets sum to
//!   0.97 (rounding, or a half-finished edit) gets a proportional rescale
//!   rather than a portfolio that is 3% cash forever.
//! - A gap below [`REBALANCE_BAND`] produces no trade. Trading to correct a
//!   rounding difference costs more in spread and tax than it recovers.

/// Trades smaller than this as a fraction of capital are suppressed.
pub const REBALANCE_BAND: f64 = 0.005;

/// One side of the gap between where a holding is and where it should be.
#[derive(Debug, Clone, PartialEq)]
pub struct RebalanceTrade {
    pub symbol: String,
    /// Weight the holding currently has.
    pub current_weight: f64,
    /// Weight it should have.
    pub target_weight: f64,
    /// `target - current`. Positive is a buy, negative is a sell.
    pub delta_weight: f64,
}

impl RebalanceTrade {
    /// Whether this gap is worth trading.
    pub fn is_material(&self) -> bool {
        self.delta_weight.abs() >= REBALANCE_BAND
    }

    /// "BUY" or "SELL".
    pub fn side(&self) -> &'static str {
        if self.delta_weight > 0.0 {
            "BUY"
        } else {
            "SELL"
        }
    }
}

/// A target allocation, normalized so the weights sum to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetWeight {
    pub symbol: String,
    /// Weight as supplied, before normalization.
    pub raw_weight: f64,
    /// Weight after the portfolio was rescaled to sum to 1.
    pub weight: f64,
}

/// Normalize `weights` to sum to 1 and pair them with their symbols.
///
/// Negative weights are dropped and a symbol with a non-finite or negative
/// weight is rejected: a short position is a different feature with different
/// risk rules, and silently folding one into a long-only rebalance would
/// produce a suggestion nobody should act on.
pub fn normalize_weights(pairs: &[(String, f64)]) -> Vec<TargetWeight> {
    let kept: Vec<(&String, f64)> = pairs
        .iter()
        .filter(|(_, w)| w.is_finite() && *w > 0.0)
        .map(|(s, w)| (s, *w))
        .collect();
    let total: f64 = kept.iter().map(|(_, w)| *w).sum();
    if total <= 0.0 || !total.is_finite() {
        return Vec::new();
    }
    kept.iter()
        .map(|(s, w)| TargetWeight {
            symbol: (*s).clone(),
            raw_weight: *w,
            weight: w / total,
        })
        .collect()
}

/// Trades that would move the book from `current_weights` to `targets`.
///
/// Both sides are normalized independently, so the caller may pass raw prices
/// or hand-computed percentages. The result is sorted by descending absolute
/// gap, so the biggest rebalance is the first thing shown.
pub fn rebalance_to_targets(
    current_weights: &[(String, f64)],
    targets: &[(String, f64)],
) -> Vec<RebalanceTrade> {
    let current = normalize_weights(current_weights);
    let target = normalize_weights(targets);

    if current.is_empty() || target.is_empty() {
        return Vec::new();
    }

    // Union of symbols on both sides: a holding absent from the targets is a
    // full exit, and a target absent from the book is a full entry.
    let mut symbols: Vec<&str> = Vec::new();
    for w in current.iter().chain(target.iter()) {
        if !symbols.contains(&w.symbol.as_str()) {
            symbols.push(&w.symbol);
        }
    }

    let lookup = |list: &[TargetWeight], s: &str| {
        list.iter()
            .find(|w| w.symbol == s)
            .map(|w| w.weight)
            .unwrap_or(0.0)
    };

    let mut trades: Vec<RebalanceTrade> = symbols
        .iter()
        .map(|s| {
            let c = lookup(&current, s);
            let t = lookup(&target, s);
            RebalanceTrade {
                symbol: (*s).to_string(),
                current_weight: c,
                target_weight: t,
                delta_weight: t - c,
            }
        })
        .filter(|t| t.is_material())
        .collect();

    trades.sort_by(|a, b| {
        b.delta_weight
            .abs()
            .partial_cmp(&a.delta_weight.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    trades
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_normalize_to_one() {
        let n = normalize_weights(&[
            ("A".into(), 60.0),
            ("B".into(), 30.0),
            ("C".into(), 10.0),
        ]);
        let sum: f64 = n.iter().map(|w| w.weight).sum();
        assert!((sum - 1.0).abs() < 1e-12, "{sum}");
        assert!((n[0].weight - 0.6).abs() < 1e-12);
        assert!((n[0].raw_weight - 60.0).abs() < 1e-12, "raw is kept");
    }

    #[test]
    fn weights_that_do_not_sum_to_one_are_rescaled() {
        // A half-finished target list must not leave permanent phantom cash.
        let n = normalize_weights(&[("A".into(), 0.5), ("B".into(), 0.47)]);
        let sum: f64 = n.iter().map(|w| w.weight).sum();
        assert!((sum - 1.0).abs() < 1e-12, "{n:?}");
    }

    #[test]
    fn negative_and_non_finite_weights_are_dropped() {
        let n = normalize_weights(&[
            ("A".into(), 1.0),
            ("SHORT".into(), -0.5),
            ("NAN".into(), f64::NAN),
            ("INF".into(), f64::INFINITY),
        ]);
        assert_eq!(n.len(), 1, "{n:?}");
        assert_eq!(n[0].symbol, "A");
    }

    #[test]
    fn all_zero_weights_yield_nothing_rather_than_dividing_by_zero() {
        assert!(normalize_weights(&[("A".into(), 0.0), ("B".into(), 0.0)]).is_empty());
        assert!(normalize_weights(&[]).is_empty());
        assert!(rebalance_to_targets(&[("A".into(), 0.0)], &[("A".into(), 1.0)]).is_empty());
    }

    #[test]
    fn a_matching_book_needs_no_trades() {
        let book = [("A".into(), 50.0), ("B".into(), 50.0)];
        assert!(
            rebalance_to_targets(&book, &book).is_empty(),
            "an already-targeted book must produce no churn"
        );
    }

    #[test]
    fn rebalancing_produces_buys_and_sells_that_offset() {
        let current = [("A".into(), 80.0), ("B".into(), 20.0)];
        let targets = [("A".into(), 50.0), ("B".into(), 50.0)];
        let trades = rebalance_to_targets(&current, &targets);
        assert_eq!(trades.len(), 2, "{trades:?}");
        // Everything must net to zero weight change, or the book would not
        // actually reach the target.
        let net: f64 = trades.iter().map(|t| t.delta_weight).sum();
        assert!(net.abs() < 1e-12, "net weight change {net} must be zero");
        let a = trades.iter().find(|t| t.symbol == "A").unwrap();
        let b = trades.iter().find(|t| t.symbol == "B").unwrap();
        assert_eq!(a.side(), "SELL");
        assert_eq!(b.side(), "BUY");
    }

    #[test]
    fn a_holding_missing_from_the_targets_is_a_full_exit() {
        let current = [("A".into(), 50.0), ("LEGACY".into(), 50.0)];
        let targets = [("A".into(), 100.0)];
        let trades = rebalance_to_targets(&current, &targets);
        let legacy = trades.iter().find(|t| t.symbol == "LEGACY").unwrap();
        assert_eq!(legacy.side(), "SELL");
        assert!((legacy.delta_weight + 0.5).abs() < 1e-12, "{legacy:?}");
    }

    #[test]
    fn a_new_target_is_a_full_entry() {
        let current = [("A".into(), 100.0)];
        let targets = [("A".into(), 70.0), ("NEW".into(), 30.0)];
        let trades = rebalance_to_targets(&current, &targets);
        let new = trades.iter().find(|t| t.symbol == "NEW").unwrap();
        assert_eq!(new.side(), "BUY");
        assert!((new.delta_weight - 0.3).abs() < 1e-12, "{new:?}");
    }

    #[test]
    fn sub_band_gaps_are_suppressed() {
        // A 0.2% drift is not worth a trade.
        let current = [("A".into(), 50.2), ("B".into(), 49.8)];
        let targets = [("A".into(), 50.0), ("B".into(), 50.0)];
        assert!(rebalance_to_targets(&current, &targets).is_empty());
    }

    #[test]
    fn trades_are_sorted_by_largest_gap_first() {
        let current = [("A".into(), 10.0), ("B".into(), 20.0), ("C".into(), 70.0)];
        let targets = [("A".into(), 60.0), ("B".into(), 25.0), ("C".into(), 15.0)];
        let trades = rebalance_to_targets(&current, &targets);
        let gaps: Vec<f64> = trades.iter().map(|t| t.delta_weight.abs()).collect();
        assert!(
            gaps.windows(2).all(|w| w[0] >= w[1] - 1e-12),
            "not sorted by gap size: {gaps:?}"
        );
        assert_eq!(trades[0].symbol, "C", "the 55pt move should lead over A's 50pt");
    }

    #[test]
    fn one_empty_side_yields_nothing() {
        assert!(rebalance_to_targets(&[], &[("A".into(), 1.0)]).is_empty());
        assert!(rebalance_to_targets(&[("A".into(), 1.0)], &[]).is_empty());
    }

    #[test]
    fn output_is_finite_for_hostile_input() {
        let trades = rebalance_to_targets(
            &[("A".into(), f64::NAN), ("B".into(), 1.0)],
            &[("A".into(), 1.0), ("B".into(), f64::INFINITY), ("C".into(), 2.0)],
        );
        for t in &trades {
            assert!(t.current_weight.is_finite(), "{t:?}");
            assert!(t.target_weight.is_finite(), "{t:?}");
            assert!(t.delta_weight.is_finite(), "{t:?}");
        }
    }

    #[test]
    fn rebalancing_is_deterministic() {
        let current = [("A".into(), 80.0), ("B".into(), 20.0)];
        let targets = [("A".into(), 50.0), ("B".into(), 50.0)];
        assert_eq!(
            rebalance_to_targets(&current, &targets),
            rebalance_to_targets(&current, &targets)
        );
    }
}
// crates/bt-analytics/src/portfolio.rs
// Author: Sourish Dey

//! Portfolio holdings, P&L, allocation and risk.
//!
//! Pure arithmetic over holdings plus a price map. Prices come from the caller
//! (cached closes in the app, fetched quotes in the CLI), so this module never
//! touches the network and every number is unit-testable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One position: quantity held at an average cost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Holding {
    pub symbol: String,
    pub qty: f64,
    pub avg_price: f64,
}

impl Holding {
    /// Market value at `price`, or 0 for nonsense input (never negative).
    pub fn market_value(&self, price: f64) -> f64 {
        if self.qty <= 0.0 || !price.is_finite() || price < 0.0 {
            return 0.0;
        }
        self.qty * price
    }

    /// Unrealised P&L at `price`.
    pub fn pnl(&self, price: f64) -> f64 {
        if self.qty <= 0.0 || !price.is_finite() {
            return 0.0;
        }
        self.qty * (price - self.avg_price)
    }

    /// Unrealised P&L as a percent of cost.
    pub fn pnl_pct(&self, price: f64) -> f64 {
        let cost = self.qty * self.avg_price;
        if cost.abs() < 1e-12 {
            return 0.0;
        }
        self.pnl(price) / cost * 100.0
    }
}

/// Cost basis of the whole book.
pub fn total_cost(holdings: &[Holding]) -> f64 {
    holdings
        .iter()
        .map(|h| (h.qty.max(0.0)) * h.avg_price.max(0.0))
        .sum()
}

/// Market value of the whole book at `prices` (missing symbols count as 0).
pub fn total_value(holdings: &[Holding], prices: &HashMap<String, f64>) -> f64 {
    holdings
        .iter()
        .map(|h| h.market_value(prices.get(&h.symbol).copied().unwrap_or(0.0)))
        .sum()
}

/// Total unrealised P&L.
pub fn compute_pnl(holdings: &[Holding], prices: &HashMap<String, f64>) -> f64 {
    holdings
        .iter()
        .map(|h| h.pnl(prices.get(&h.symbol).copied().unwrap_or(0.0)))
        .sum()
}

/// Weight of each holding by market value, in percent. Sums to 100 unless the
/// book is worth nothing, in which case every weight is 0.
pub fn allocation(holdings: &[Holding], prices: &HashMap<String, f64>) -> Vec<(String, f64)> {
    let total = total_value(holdings, prices);
    holdings
        .iter()
        .map(|h| {
            let w = if total > 0.0 {
                h.market_value(prices.get(&h.symbol).copied().unwrap_or(0.0)) / total * 100.0
            } else {
                0.0
            };
            (h.symbol.clone(), w)
        })
        .collect()
}

/// Book-level risk: portfolio volatility from per-symbol daily-return series.
///
/// `returns` maps symbol → daily returns; weights come from current market
/// values. Assumes zero cross-correlation (a stated, conservative-for-diversification
/// simplification documented here rather than hidden): variance is the
/// weight-squared sum of variances, annualised by √252.
pub fn risk_metrics(
    holdings: &[Holding],
    prices: &HashMap<String, f64>,
    returns: &HashMap<String, Vec<f64>>,
) -> PortfolioRisk {
    let total = total_value(holdings, prices).max(1e-12);
    let mut var_sum = 0.0;
    let mut worst_dd = 0.0f64;
    for h in holdings {
        let w = h.market_value(prices.get(&h.symbol).copied().unwrap_or(0.0)) / total;
        if let Some(rs) = returns.get(&h.symbol) {
            if rs.len() > 1 {
                let mean = rs.iter().sum::<f64>() / rs.len() as f64;
                let var =
                    rs.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rs.len() - 1) as f64;
                var_sum += w * w * var.max(0.0);
                // Worst single-day fall among holdings, as a rough stress read.
                let worst = rs.iter().cloned().fold(f64::MAX, f64::min);
                worst_dd = worst_dd.min(worst * w);
            }
        }
    }
    PortfolioRisk {
        annual_vol_pct: var_sum.sqrt() * (252.0_f64).sqrt() * 100.0,
        worst_weighted_day_pct: worst_dd * 100.0,
    }
}

/// Book-level risk readouts.
#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioRisk {
    /// Annualised volatility, percent.
    pub annual_vol_pct: f64,
    /// Worst single-day weighted fall, percent (negative or zero).
    pub worst_weighted_day_pct: f64,
}

/// Validate holdings as loaded from `portfolio.json`: non-empty symbols,
/// non-negative quantities and prices.
pub fn validate(holdings: &[Holding]) -> Result<(), String> {
    for (i, h) in holdings.iter().enumerate() {
        if h.symbol.trim().is_empty() {
            return Err(format!("holding {i}: empty symbol"));
        }
        // NaN is spelled out: `qty <= 0.0` alone would let a NaN quantity pass
        // validation it cannot satisfy.
        if h.qty.is_nan() || h.qty <= 0.0 {
            return Err(format!("holding {}: bad qty {}", h.symbol, h.qty));
        }
        if h.avg_price.is_nan() || h.avg_price < 0.0 {
            return Err(format!("holding {}: bad avg_price {}", h.symbol, h.avg_price));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> Vec<Holding> {
        vec![
            Holding { symbol: "A".into(), qty: 10.0, avg_price: 100.0 },
            Holding { symbol: "B".into(), qty: 5.0, avg_price: 200.0 },
        ]
    }

    fn prices() -> HashMap<String, f64> {
        [("A".to_string(), 110.0), ("B".to_string(), 190.0)]
            .into_iter()
            .collect()
    }

    #[test]
    fn pnl_and_allocation_add_up() {
        let (b, p) = (book(), prices());
        assert_eq!(total_cost(&b), 2000.0);
        // A: 10×110=1100, B: 5×190=950 → 2050.
        assert_eq!(total_value(&b, &p), 2050.0);
        // A: +100, B: −50 → +50.
        assert_eq!(compute_pnl(&b, &p), 50.0);
        let alloc = allocation(&b, &p);
        let sum: f64 = alloc.iter().map(|(_, w)| w).sum();
        assert!((sum - 100.0).abs() < 1e-9);
        assert!(alloc[0].1 > alloc[1].1);
    }

    #[test]
    fn missing_prices_count_as_zero_not_nan() {
        let b = book();
        let empty: HashMap<String, f64> = HashMap::new();
        assert_eq!(total_value(&b, &empty), 0.0);
        assert_eq!(compute_pnl(&b, &empty), -2000.0);
        assert!(allocation(&b, &empty).iter().all(|(_, w)| *w == 0.0));
    }

    #[test]
    fn risk_scales_with_weights() {
        let (b, p) = (book(), prices());
        let returns: HashMap<String, Vec<f64>> = [
            ("A".to_string(), vec![0.01, -0.01, 0.01, -0.01]),
            ("B".to_string(), vec![0.0; 4]),
        ]
        .into_iter()
        .collect();
        let r = risk_metrics(&b, &p, &returns);
        assert!(r.annual_vol_pct > 0.0);
        assert!(r.worst_weighted_day_pct <= 0.0);
    }

    #[test]
    fn validation_names_the_bad_row() {
        assert!(validate(&book()).is_ok());
        assert!(validate(&[Holding { symbol: "".into(), qty: 1.0, avg_price: 1.0 }]).is_err());
        assert!(validate(&[Holding { symbol: "A".into(), qty: -1.0, avg_price: 1.0 }]).is_err());
    }

    #[test]
    fn holdings_round_trip_through_json() {
        let json = serde_json::to_string(&book()).unwrap();
        let back: Vec<Holding> = serde_json::from_str(&json).unwrap();
        assert_eq!(book(), back);
    }
}

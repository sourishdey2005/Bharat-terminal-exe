// crates/bt-analytics/src/options.rs
// Author: Sourish Dey

//! Option math over [`OptionChain`]: Black-Scholes Greeks,
//! put-call ratios and max pain.
//!
//! Everything here is pure arithmetic on numbers the feeds already supplied.
//! NSE legs carry exchange-published IVs; Yahoo legs carry IVs too, so this
//! module prices *sensitivities*, never the premiums themselves. Conventions:
//! theta is per calendar day, vega and rho per one point of vol/rate (×100
//! from the textbook per-unit form), matching how the chain table displays them.

use bt_core::OptionChain;

/// Black-Scholes sensitivities for one European option.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    /// Per day.
    pub theta: f64,
    /// Per 1 vol point.
    pub vega: f64,
    /// Per 1 rate point.
    pub rho: f64,
}

/// Standard normal CDF via the Abramowitz–Stegun 7.1.26 approximation
/// (absolute error < 7.5e-8, far below any IV quote precision).
fn norm_cdf(x: f64) -> f64 {
    if x >= 6.0 {
        return 1.0;
    }
    if x <= -6.0 {
        return 0.0;
    }
    let (sign, x) = if x < 0.0 { (-1.0, -x) } else { (1.0, x) };
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t
        - 0.356563782) * t
        + 0.319381530)
        * t;
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * poly;
    if sign < 0.0 {
        1.0 - cdf
    } else {
        cdf
    }
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Black-Scholes Greeks.
///
/// `time_to_expiry` in years (days/365), `risk_free` and `volatility` as
/// decimals (0.065, 0.20). Degenerate inputs (non-positive spot/strike/vol or
/// non-positive time) yield zero Greeks rather than NaN, because a NaN delta
/// in the chain table would read as a quote.
pub fn black_scholes_greeks(
    spot: f64,
    strike: f64,
    time_to_expiry: f64,
    risk_free: f64,
    volatility: f64,
    is_call: bool,
) -> Greeks {
    // Degenerate inputs yield zeros rather than NaN. Each check admits NaN
    // explicitly: `volatility <= 0.0` alone would let a NaN vol through
    // (NaN comparisons are false), handing the pricer a poisoned input.
    let sane = !spot.is_nan()
        && spot > 0.0
        && !strike.is_nan()
        && strike > 0.0
        && !volatility.is_nan()
        && volatility > 0.0
        && !time_to_expiry.is_nan()
        && time_to_expiry > 0.0;
    if !sane {
        return Greeks {
            delta: 0.0,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
        };
    }
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (risk_free + 0.5 * volatility * volatility) * time_to_expiry)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let pdf_d1 = norm_pdf(d1);
    let discount = (-risk_free * time_to_expiry).exp();

    let delta = if is_call {
        nd1
    } else {
        nd1 - 1.0
    };
    let gamma = pdf_d1 / (spot * volatility * sqrt_t);
    let theta_annual = -(spot * pdf_d1 * volatility / (2.0 * sqrt_t))
        + if is_call {
            -risk_free * strike * discount * nd2
        } else {
            risk_free * strike * discount * (1.0 - nd2)
        };
    let vega = spot * pdf_d1 * sqrt_t * 0.01;
    let rho = if is_call {
        strike * time_to_expiry * discount * nd2 * 0.01
    } else {
        -strike * time_to_expiry * discount * (1.0 - nd2) * 0.01
    };
    Greeks {
        delta,
        gamma,
        theta: theta_annual / 365.0,
        vega,
        rho,
    }
}

/// Total put OI / total call OI. Above 1 leans bearish positioning.
pub fn pcr_oi(chain: &OptionChain) -> f64 {
    let (mut puts, mut calls) = (0u64, 0u64);
    for r in &chain.strikes {
        puts += r.put.oi;
        calls += r.call.oi;
    }
    if calls == 0 {
        return 0.0;
    }
    puts as f64 / calls as f64
}

/// Total put volume / total call volume.
pub fn pcr_volume(chain: &OptionChain) -> f64 {
    let (mut puts, mut calls) = (0u64, 0u64);
    for r in &chain.strikes {
        puts += r.put.volume;
        calls += r.call.volume;
    }
    if calls == 0 {
        return 0.0;
    }
    puts as f64 / calls as f64
}

/// Strike nearest the underlying (prefers the row already tagged ATM).
pub fn atm_strike(chain: &OptionChain) -> f64 {
    if let Some(row) = chain.strikes.iter().find(|r| r.atm) {
        return row.strike;
    }
    if chain.underlying_value <= 0.0 || chain.strikes.is_empty() {
        return 0.0;
    }
    chain
        .strikes
        .iter()
        .min_by(|a, b| {
            (a.strike - chain.underlying_value)
                .abs()
                .partial_cmp(&(b.strike - chain.underlying_value).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.strike)
        .unwrap_or(0.0)
}

/// Max-pain strike: the expiry price minimizing total writer payout.
///
/// For each candidate strike, every ITM call pays (candidate − strike) × OI
/// and every ITM put pays (strike − candidate) × OI. The minimum-total-loss
/// strike is where writers lose least, the level the market is incentivised
/// to pin. Returns 0.0 on an empty chain rather than guessing.
pub fn max_pain(chain: &OptionChain) -> f64 {
    if chain.strikes.is_empty() {
        return 0.0;
    }
    let mut best = (chain.strikes[0].strike, f64::MAX);
    for cand in &chain.strikes {
        let px = cand.strike;
        let mut loss = 0.0;
        for r in &chain.strikes {
            if px > r.strike {
                loss += (px - r.strike) * r.call.oi as f64;
            }
            if px < r.strike {
                loss += (r.strike - px) * r.put.oi as f64;
            }
        }
        if loss < best.1 {
            best = (px, loss);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use bt_core::{OptionChain, OptionLeg, OptionStrike};

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// S=100 K=100 T=1 r=0.05 σ=0.20: d1=0.35, N(d1)≈0.6368, N(d2)≈0.5596.
    /// Hand-computed from the closed form, not from another library.
    #[test]
    fn atm_call_greeks_match_the_closed_form() {
        let g = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.20, true);
        assert!(approx(g.delta, 0.6368, 1e-3), "delta {}", g.delta);
        assert!(approx(g.gamma, 0.0188, 1e-3), "gamma {}", g.gamma);
        assert!(approx(g.vega, 0.3752, 1e-3), "vega {}", g.vega);
        assert!(approx(g.rho, 0.5323, 1e-3), "rho {}", g.rho);
        // Call theta ≈ -(3.752 + 2.659)/365 per day.
        assert!(approx(g.theta, -0.0176, 1e-3), "theta {}", g.theta);
    }

    #[test]
    fn put_call_parity_holds_on_delta() {
        let c = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.20, true);
        let p = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.20, false);
        assert!(approx(c.delta - p.delta, 1.0, 1e-9));
        assert!(approx(c.gamma, p.gamma, 1e-12), "gamma is strike-symmetric");
        assert!(approx(c.vega, p.vega, 1e-12), "vega is strike-symmetric");
    }

    #[test]
    fn deep_itm_and_otm_deltas_saturate() {
        let deep_call = black_scholes_greeks(200.0, 100.0, 1.0, 0.05, 0.20, true);
        assert!(deep_call.delta > 0.99);
        let far_put = black_scholes_greeks(200.0, 100.0, 1.0, 0.05, 0.20, false);
        assert!(far_put.delta.abs() < 0.01);
    }

    #[test]
    fn degenerate_inputs_yield_zeros_not_nan() {
        for g in [
            black_scholes_greeks(0.0, 100.0, 1.0, 0.05, 0.20, true),
            black_scholes_greeks(100.0, 0.0, 1.0, 0.05, 0.20, true),
            black_scholes_greeks(100.0, 100.0, 0.0, 0.05, 0.20, true),
            black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.0, true),
            black_scholes_greeks(f64::NAN, 100.0, 1.0, 0.05, 0.20, true),
        ] {
            assert_eq!(g.delta, 0.0);
            assert_eq!(g.gamma, 0.0);
            assert_eq!(g.theta, 0.0);
            assert_eq!(g.vega, 0.0);
            assert_eq!(g.rho, 0.0);
        }
    }

    fn chain_fixture() -> OptionChain {
        // Calls heavy at 100, puts heavy at 102: writers lose least at 101.
        let leg = |oi: u64| OptionLeg {
            oi,
            oi_change: 0,
            volume: oi / 2,
            ltp: 1.0,
            iv: 0.2,
        };
        OptionChain {
            symbol: "TEST".into(),
            expiry: "30-Oct-2026".into(),
            underlying_value: 101.0,
            strikes: vec![
                OptionStrike {
                    strike: 100.0,
                    call: leg(1000),
                    put: leg(100),
                    atm: false,
                },
                OptionStrike {
                    strike: 101.0,
                    call: leg(100),
                    put: leg(100),
                    atm: true,
                },
                OptionStrike {
                    strike: 102.0,
                    call: leg(100),
                    put: leg(1000),
                    atm: false,
                },
            ],
            fetched_at: 0,
        }
    }

    #[test]
    fn pcr_reflects_positioning() {
        // Puts: 100+100+1000=1200. Calls: 1000+100+100=1200. PCR = 1.
        assert!(approx(pcr_oi(&chain_fixture()), 1.0, 1e-9));
        assert!(approx(pcr_volume(&chain_fixture()), 1.0, 1e-9));
        let mut one_sided = chain_fixture();
        for r in &mut one_sided.strikes {
            r.put.oi = 0;
            r.put.volume = 0;
        }
        assert_eq!(pcr_oi(&one_sided), 0.0);
        assert_eq!(pcr_volume(&one_sided), 0.0);
    }

    #[test]
    fn max_pain_finds_the_writer_minimum() {
        // Loss at 100: puts pay (101-100)*100 + (102-100)*1000 = 2100.
        // Loss at 101: calls pay (101-100)*1000=1000, puts pay (102-101)*1000=1000 → 2000.
        // Loss at 102: calls pay (102-100)*1000 + (102-101)*100 = 2100.
        assert_eq!(max_pain(&chain_fixture()), 101.0);
    }

    #[test]
    fn atm_prefers_the_tagged_row() {
        assert_eq!(atm_strike(&chain_fixture()), 101.0);
        let mut untagged = chain_fixture();
        for r in &mut untagged.strikes {
            r.atm = false;
        }
        assert_eq!(atm_strike(&untagged), 101.0);
        let empty = OptionChain {
            symbol: "X".into(),
            expiry: String::new(),
            underlying_value: 0.0,
            strikes: vec![],
            fetched_at: 0,
        };
        assert_eq!(max_pain(&empty), 0.0);
        assert_eq!(atm_strike(&empty), 0.0);
        assert_eq!(pcr_oi(&empty), 0.0);
    }
}

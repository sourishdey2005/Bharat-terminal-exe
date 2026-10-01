// crates/bt-analytics/src/screener.rs
// Author: Sourish Dey

//! Technical stock screener.
//!
//! Filters are plain-text conjunctions like `rsi<30 AND change_5d>2`, parsed by
//! [`parse_filters`] and applied by [`apply_filters`]. Only technical fields
//! computed from price/volume exist here — fundamental fields (`pe`, `pb`,
//! `roe`, …) fail parsing with a message that says so, rather than silently
//! matching nothing. Supported fields are listed on [`TECHNICAL_FIELDS`].

use std::collections::HashMap;

/// Technical fields `compute_technicals` produces. Anything else is rejected
/// at parse time with the field name in the error.
pub const TECHNICAL_FIELDS: &[&str] = &[
    "price",
    "rsi",
    "macd_hist",
    "change_5d",
    "change_20d",
    "volume_ratio",
    "sma50_ratio",
    "sma200_ratio",
    "bb_pos",
    "high52_dist",
    "low52_dist",
];

/// One row of screenable metrics for a symbol.
#[derive(Debug, Clone)]
pub struct ScreenRow {
    pub symbol: String,
    pub metrics: HashMap<String, f64>,
}

impl ScreenRow {
    pub fn get(&self, field: &str) -> Option<f64> {
        self.metrics.get(field).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    LtEq,
    Gt,
    GtEq,
    Eq,
}

/// One parsed `field op value` clause.
#[derive(Debug, Clone)]
pub struct Filter {
    pub field: String,
    pub op: CmpOp,
    pub value: f64,
}

/// Parse `rsi<30 AND change_5d>2`. Clauses split on case-insensitive `AND`;
/// each clause is `field`, one of `< <= > >= ==` (longest match first), and a
/// number. Unknown fields and malformed clauses are errors naming the problem.
pub fn parse_filters(s: &str) -> Result<Vec<Filter>, String> {
    let mut out = Vec::new();
    // Split on AND without pulling in a regex dependency: walk case-insensitively.
    let upper = s.to_uppercase();
    let mut clauses = Vec::new();
    let mut start = 0usize;
    let bytes = upper.as_bytes();
    let mut i = 0usize;
    while i + 3 <= bytes.len() {
        if &bytes[i..i + 3] == b"AND"
            && (i == 0 || bytes[i - 1].is_ascii_whitespace())
            && (i + 3 == bytes.len() || bytes[i + 3].is_ascii_whitespace())
        {
            clauses.push(s[start..i].trim());
            i += 3;
            start = i;
        } else {
            i += 1;
        }
    }
    clauses.push(s[start..].trim());
    for clause in clauses.into_iter().filter(|c| !c.is_empty()) {
        out.push(parse_clause(clause)?);
    }
    if out.is_empty() {
        return Err("empty filter".to_string());
    }
    Ok(out)
}

fn parse_clause(clause: &str) -> Result<Filter, String> {
    for op in ["<=", ">=", "==", "<", ">"] {
        if let Some(pos) = clause.find(op) {
            let field = clause[..pos].trim().to_lowercase();
            let value: f64 = clause[pos + op.len()..].trim().parse().map_err(|_| {
                format!("filter {clause:?}: value is not a number")
            })?;
            if !TECHNICAL_FIELDS.contains(&field.as_str()) {
                return Err(format!(
                    "unknown filter field {field:?}; technical fields are: {}",
                    TECHNICAL_FIELDS.join(", ")
                ));
            }
            let op = match op {
                "<" => CmpOp::Lt,
                "<=" => CmpOp::LtEq,
                ">" => CmpOp::Gt,
                ">=" => CmpOp::GtEq,
                _ => CmpOp::Eq,
            };
            return Ok(Filter { field, op, value });
        }
    }
    Err(format!("filter {clause:?}: expected e.g. rsi<30"))
}

/// Keep rows matching every clause.
pub fn apply_filters<'a>(rows: &'a [ScreenRow], filters: &[Filter]) -> Vec<&'a ScreenRow> {
    rows.iter()
        .filter(|r| {
            filters.iter().all(|f| match r.get(&f.field) {
                None => false,
                Some(v) => match f.op {
                    CmpOp::Lt => v < f.value,
                    CmpOp::LtEq => v <= f.value,
                    CmpOp::Gt => v > f.value,
                    CmpOp::GtEq => v >= f.value,
                    CmpOp::Eq => (v - f.value).abs() <= 1e-9,
                },
            })
        })
        .collect()
}

/// Compute the technical metrics for one symbol from closes + volumes.
///
/// Short histories yield NaN metrics rather than errors; `apply_filters` treats
/// NaN as "does not match", so thin symbols are skipped instead of crashing a
/// 300-symbol screen.
pub fn compute_technicals(symbol: &str, closes: &[f64], volumes: &[f64]) -> ScreenRow {
    use crate::indicators::{bollinger, macd, rsi};
    use bt_core::{Candle, OhlcvSeries};

    let mut metrics = HashMap::new();
    let n = closes.len();
    if n == 0 {
        return ScreenRow {
            symbol: symbol.to_string(),
            metrics,
        };
    }
    let last = closes[n - 1];
    metrics.insert("price".to_string(), last);

    let candles: Vec<Candle> = closes
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let v = volumes.get(i).copied().unwrap_or(0.0);
            Candle::new(i as f64, c, c, c, c, v)
        })
        .collect();
    let series = OhlcvSeries::new(symbol, candles);

    let val = |v: &[f64]| v.iter().rev().find(|x| x.is_finite()).copied().unwrap_or(f64::NAN);
    metrics.insert("rsi".to_string(), val(&rsi(&series, 14)));
    let (ml, sl, _) = macd(&series);
    let mh = val(&ml) - val(&sl);
    metrics.insert("macd_hist".to_string(), mh);

    let pct = |bars: usize| -> f64 {
        if n > bars {
            let base = closes[n - 1 - bars].abs().max(1e-12);
            (last / base - 1.0) * 100.0
        } else {
            f64::NAN
        }
    };
    metrics.insert("change_5d".to_string(), pct(5));
    metrics.insert("change_20d".to_string(), pct(20));

    let vol_ratio = if volumes.len() >= 21 {
        let base: f64 = volumes[volumes.len() - 21..volumes.len() - 1].iter().sum::<f64>() / 20.0;
        if base > 0.0 {
            volumes[volumes.len() - 1] / base
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };
    metrics.insert("volume_ratio".to_string(), vol_ratio);

    let sma = |w: usize| -> f64 {
        if n >= w {
            closes[n - w..].iter().sum::<f64>() / w as f64
        } else {
            f64::NAN
        }
    };
    let (s50, s200) = (sma(50), sma(200));
    metrics.insert(
        "sma50_ratio".to_string(),
        if s50.is_finite() && s50.abs() > 1e-12 {
            last / s50
        } else {
            f64::NAN
        },
    );
    metrics.insert(
        "sma200_ratio".to_string(),
        if s200.is_finite() && s200.abs() > 1e-12 {
            last / s200
        } else {
            f64::NAN
        },
    );

    let (mid, upper, lower) = bollinger(&series, 20, 2.0);
    let (lo, _m, hi) = (
        val(&lower),
        val(&mid),
        val(&upper),
    );
    metrics.insert(
        "bb_pos".to_string(),
        if hi.is_finite() && lo.is_finite() && (hi - lo).abs() > 1e-12 {
            (last - lo) / (hi - lo)
        } else {
            f64::NAN
        },
    );

    let w52 = n.min(252);
    let hi52 = closes[n - w52..].iter().cloned().fold(f64::MIN, f64::max);
    let lo52 = closes[n - w52..].iter().cloned().fold(f64::MAX, f64::min);
    metrics.insert(
        "high52_dist".to_string(),
        if hi52 > 0.0 { (last - hi52) / hi52 * 100.0 } else { f64::NAN },
    );
    metrics.insert(
        "low52_dist".to_string(),
        if lo52 > 0.0 { (last - lo52) / lo52 * 100.0 } else { f64::NAN },
    );

    ScreenRow {
        symbol: symbol.to_string(),
        metrics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rising(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64).collect()
    }

    #[test]
    fn parses_conjunctions_and_all_operators() {
        let fs = parse_filters("rsi<30 AND change_5d>=2 AND price==100").unwrap();
        assert_eq!(fs.len(), 3);
        assert_eq!(fs[0].field, "rsi");
        assert_eq!(fs[0].op, CmpOp::Lt);
        assert_eq!(fs[1].op, CmpOp::GtEq);
        assert_eq!(fs[2].op, CmpOp::Eq);
    }

    #[test]
    fn unknown_fields_and_garbage_are_named_errors() {
        let err = parse_filters("pe<20").unwrap_err();
        assert!(err.contains("pe"), "{err}");
        assert!(err.contains("rsi"), "{err}");
        assert!(parse_filters("rsi<<30").is_err());
        assert!(parse_filters("rsi<abc").is_err());
        assert!(parse_filters("").is_err());
        assert!(parse_filters("rsi").is_err());
    }

    #[test]
    fn rising_series_passes_momentum_and_fails_oversold() {
        let row = compute_technicals("X", &rising(300), &vec![1000.0; 300]);
        assert_eq!(row.get("price"), Some(399.0));
        assert!(row.get("rsi").unwrap() > 70.0);
        assert!(row.get("change_5d").unwrap() > 0.0);
        assert!(row.get("sma50_ratio").unwrap() > 1.0);
        let oversold = parse_filters("rsi<30").unwrap();
        assert!(apply_filters(std::slice::from_ref(&row), &oversold).is_empty());
        let momentum = parse_filters("rsi>70 AND change_5d>0").unwrap();
        assert_eq!(apply_filters(&[row], &momentum).len(), 1);
    }

    #[test]
    fn thin_history_yields_nan_not_panic() {
        let row = compute_technicals("Y", &[100.0, 101.0], &[10.0, 12.0]);
        assert!(row.get("rsi").unwrap().is_nan());
        assert!(apply_filters(&[row], &parse_filters("rsi<30").unwrap()).is_empty());
    }

    #[test]
    fn empty_series_is_safe() {
        let row = compute_technicals("Z", &[], &[]);
        assert!(row.metrics.is_empty());
    }
}

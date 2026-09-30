// crates/bt-analytics/src/models/skill.rs
// Author: Sourish Dey

//! What each shipped model actually measured, and whether it has any edge.
//!
//! # Why this exists
//!
//! `models/forecast_models.json` is written by `train_forecasts.py` and records
//! each model's validation RMSE next to the random-walk RMSE on the *same* split.
//! That pair is the only honest answer to "is this engine worth looking at".
//!
//! It matters because of a specific failure. Windows are anchored on the last
//! close, so predicting **zero** *is* a random walk. `train_forecasts.py` seeds
//! its best-model search with the untrained epoch-0 state, whose output is
//! approximately zero, and only aborts when the final RMSE is strictly *worse*
//! than the baseline. Equality therefore passes the guard: a model that learned
//! nothing at all is exported, denormalises to "the last close, repeated", and
//! the app draws a confident flat line that is indistinguishable from a forecast.
//!
//! The shipped `dlinear.onnx` and `nhits_small.onnx` are exactly that case —
//! their recorded RMSE equals the random-walk RMSE to all 16 recorded digits.
//! Reading the pair here lets the UI say so plainly instead of implying the line
//! came from a working model.
//!
//! # Reading the measurement
//!
//! `edge` is the fractional RMSE improvement over a random walk on the training
//! split. Zero means "no better than assuming the price does not move".
//! A model with no measurable edge is not *broken* — it is telling the truth
//! about a dataset with no learnable signal at that horizon — but presenting its
//! output as a projection is not.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// File name of the sidecar written by the training script.
pub const SKILL_FILE: &str = "forecast_models.json";

/// Edge below which an engine is reported as having no measurable signal.
pub const MIN_EDGE: f64 = 0.01;

/// A model's measured validation performance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelSkill {
    /// Validation RMSE on the held-out split.
    pub val_rmse: f64,
    /// RMSE of a random walk on that same split.
    pub random_walk_rmse: f64,
    /// Input length the model expects.
    pub lookback: usize,
    /// Steps the model can emit.
    pub horizon: usize,
}

impl ModelSkill {
    /// Fractional improvement over a random walk.
    ///
    /// Positive means the model beat "the price does not move" by that fraction
    /// of RMSE. Zero or negative means it did not.
    pub fn edge(&self) -> f64 {
        if self.random_walk_rmse <= 0.0 || !self.random_walk_rmse.is_finite() {
            return 0.0;
        }
        1.0 - (self.val_rmse / self.random_walk_rmse)
    }

    /// Whether the model beat a random walk by a margin worth calling an edge.
    pub fn has_edge(&self) -> bool {
        self.edge() > MIN_EDGE
    }

    /// One-line summary for a tooltip or an inline note.
    pub fn summary(&self) -> String {
        format!(
            "validation RMSE {:.4} vs random walk {:.4} ({:+.2}% edge)",
            self.val_rmse,
            self.random_walk_rmse,
            self.edge() * 100.0
        )
    }
}

/// The measured skill of every model that has a sidecar entry.
#[derive(Debug, Clone, Default)]
pub struct SkillBook {
    by_file: HashMap<String, ModelSkill>,
}

impl SkillBook {
    /// Read the sidecar from a `models` directory.
    ///
    /// Returns an empty book when the file is absent or malformed. That is the
    /// right default: a missing measurement must not be reported as a measured
    /// failure, and it must not stop the app from forecasting.
    pub fn load(models_dir: &Path) -> Self {
        let path: PathBuf = models_dir.join(SKILL_FILE);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        Self::from_json(&text)
    }

    /// Parse the sidecar's JSON. Exposed for tests.
    pub fn from_json(text: &str) -> Self {
        let Ok(raw) = serde_json::Value::from_str(text) else {
            return Self::default();
        };
        let Some(obj) = raw.as_object() else {
            return Self::default();
        };
        let mut by_file = HashMap::new();
        for (file, entry) in obj {
            let (Some(val_rmse), Some(walk)) = (
                entry.get("val_rmse").and_then(serde_json::Value::as_f64),
                entry
                    .get("val_rmse_random_walk")
                    .and_then(serde_json::Value::as_f64),
            ) else {
                continue;
            };
            if !val_rmse.is_finite() || !walk.is_finite() {
                continue;
            }
            by_file.insert(
                file.clone(),
                ModelSkill {
                    val_rmse,
                    random_walk_rmse: walk,
                    lookback: entry
                        .get("lookback")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as usize,
                    horizon: entry
                        .get("horizon")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as usize,
                },
            );
        }
        Self { by_file }
    }

    /// Skill for a model file name such as `dlinear.onnx`.
    pub fn for_file(&self, file: &str) -> Option<&ModelSkill> {
        self.by_file.get(file)
    }

    /// Whether any model in the book claims a real edge.
    pub fn any_has_edge(&self) -> bool {
        self.by_file.values().any(ModelSkill::has_edge)
    }

    /// True when nothing in the book claims an edge.
    pub fn is_empty(&self) -> bool {
        self.by_file.is_empty()
    }
}

/// Whether a forecast's own output carries any movement at all.
///
/// Independent of the sidecar: this inspects what the engine just returned, so a
/// model file swapped in without updating `forecast_models.json` is still caught.
pub fn is_degenerate(values: &[f64], anchor: f64) -> bool {
    if values.len() < 2 || !anchor.is_finite() || anchor.abs() < 1e-12 {
        return false;
    }
    let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Below a hundredth of a percent of the price level the line is flat for
    // every practical purpose and indistinguishable from "no forecast".
    (hi - lo).abs() / anchor.abs() < 1e-4
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDECAR: &str = r#"{
      "dlinear.onnx": {
        "lookback": 32,
        "horizon": 5,
        "normalization": "last_close_anchored",
        "val_rmse": 1.0863230228424072,
        "val_rmse_random_walk": 1.0863230228424072
      },
      "nhits_small.onnx": {
        "lookback": 32,
        "horizon": 5,
        "val_rmse": 1.0863230228424072,
        "val_rmse_random_walk": 1.0863230228424072
      }
    }"#;

    /// The exact shipped state: both models identical to the baseline.
    #[test]
    fn edge_is_zero_for_a_random_walk_clone() {
        let book = SkillBook::from_json(SIDECAR);
        for file in ["dlinear.onnx", "nhits_small.onnx"] {
            let s = book.for_file(file).expect("entry");
            assert!(
                s.edge().abs() < 1e-12,
                "{file} must report zero edge, got {}",
                s.edge()
            );
            assert!(!s.has_edge(), "{file} must not claim an edge");
        }
        assert!(!book.any_has_edge());
    }

    #[test]
    fn a_real_improvement_is_reported_as_an_edge() {
        let book = SkillBook::from_json(
            r#"{"m.onnx": {"lookback": 8, "horizon": 4,
                 "val_rmse": 0.9, "val_rmse_random_walk": 1.2}}"#,
        );
        let s = book.for_file("m.onnx").expect("entry");
        assert!((s.edge() - 0.25).abs() < 1e-12, "edge was {}", s.edge());
        assert!(s.has_edge());
        assert!(book.any_has_edge());
        assert_eq!(s.lookback, 8);
        assert_eq!(s.horizon, 4);
        assert!(s.summary().contains("edge"));
    }

    #[test]
    fn a_worse_than_baseline_model_is_not_an_edge() {
        let book =
            SkillBook::from_json(r#"{"m.onnx": {"val_rmse": 1.5, "val_rmse_random_walk": 1.2}}"#);
        let s = book.for_file("m.onnx").expect("entry");
        assert!(s.edge() < 0.0);
        assert!(!s.has_edge());
    }

    /// A missing or broken sidecar must degrade quietly, never invent a verdict.
    #[test]
    fn a_missing_sidecar_is_not_a_failure() {
        assert!(SkillBook::load(Path::new("definitely-not-here")).is_empty());
        assert!(SkillBook::from_json("not json at all").is_empty());
        assert!(SkillBook::from_json("[1,2,3]").is_empty());
        assert!(SkillBook::from_json(r#"{"m.onnx": {"lookback": 8}}"#).is_empty());
        // A zero baseline cannot be divided by; it must report zero, not NaN.
        let z =
            SkillBook::from_json(r#"{"m.onnx": {"val_rmse": 0.0, "val_rmse_random_walk": 0.0}}"#);
        assert_eq!(z.for_file("m.onnx").expect("entry").edge(), 0.0);
    }

    #[test]
    fn degenerate_detection_catches_a_repeated_anchor() {
        // Exactly what DLinear and N-HiTS emit today.
        assert!(is_degenerate(&[46990.0; 5], 46990.0));
        // A real forecast is not degenerate even if it barely moves.
        assert!(!is_degenerate(&[46990.0, 47010.0, 47005.0], 46990.0));
        // Too short to judge, or a nonsense anchor, must not accuse.
        assert!(!is_degenerate(&[1.0], 1.0));
        assert!(!is_degenerate(&[1.0; 5], 0.0));
        assert!(!is_degenerate(&[1.0; 5], f64::NAN));
    }

    #[test]
    fn a_tenth_of_a_percent_of_movement_is_not_flat() {
        // ~47 points on 46,990 is clearly a forecast, not a repeat of the anchor.
        assert!(!is_degenerate(&[46990.0, 47037.0], 46990.0));
    }
}

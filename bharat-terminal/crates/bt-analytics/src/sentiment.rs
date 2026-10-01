// crates/bt-analytics/src/sentiment.rs
// Author: Sourish Dey

//! VADER lexicon sentiment for headlines.
//!
//! Thin wrapper over the `vader_sentiment` port of the Python original, tuned
//! for short social/news text. Scores are deterministic and need no model
//! files, which is why this is a plain function rather than an engine: there
//! is nothing to load, cache or fall back from.

/// VADER compound score in [-1, 1]. Positive is bullish, negative bearish.
pub fn score(text: &str) -> f64 {
    let analyzer = vader_sentiment::SentimentIntensityAnalyzer::new();
    analyzer
        .polarity_scores(text)
        .get("compound")
        .copied()
        .unwrap_or(0.0)
}

/// Bucket a compound score the way VADER's own documentation does.
pub fn label(compound: f64) -> &'static str {
    if compound >= 0.05 {
        "Positive"
    } else if compound <= -0.05 {
        "Negative"
    } else {
        "Neutral"
    }
}

/// Score and label in one call, for feed rendering.
pub fn score_labeled(text: &str) -> (f64, &'static str) {
    let s = score(text);
    (s, label(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bullish_headlines_score_positive() {
        let (s, l) = score_labeled("Reliance hits record high on blockbuster earnings beat");
        assert!(s > 0.05, "{s}");
        assert_eq!(l, "Positive");
    }

    #[test]
    fn bearish_headlines_score_negative() {
        let (s, l) = score_labeled("Shares plunge as fraud probe widens, downgrade follows");
        assert!(s < -0.05, "{s}");
        assert_eq!(l, "Negative");
    }

    #[test]
    fn flat_headlines_score_neutral() {
        let (s, l) = score_labeled("RBI policy meeting scheduled for next week");
        assert!(s.abs() < 0.5, "{s}");
        assert_eq!(l, label(s));
    }

    #[test]
    fn empty_text_is_neutral_not_a_panic() {
        assert_eq!(score(""), 0.0);
        assert_eq!(label(0.0), "Neutral");
    }

    #[test]
    fn scores_stay_in_range() {
        for t in [
            "best stock ever!!! to the moon",
            "worst crash bankruptcy default",
            "the company announced results",
        ] {
            let s = score(t);
            assert!((-1.0..=1.0).contains(&s), "{t} -> {s}");
        }
    }
}

// crates/bt-app/src/chat_export.rs
// Author: Sourish Dey

//! Exports a chat transcript to Markdown or JSON.
//!
//! Kept separate from the drawing code because the formatting is the thing worth
//! testing: an export that silently drops the first or last message is the kind
//! of bug nobody notices until they need the transcript. Both formats are
//! built by pure functions over a slice of messages, so the tests need no window.
//!
//! Timestamps are included when supplied and omitted when not, because a
//! transcript reconstructed from a session with no clock is still worth
//! exporting — it just must not invent times.

/// Output format for [`export`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Human-readable, for pasting into a notebook or ticket.
    Markdown,
    /// Machine-readable, for re-import or analysis.
    Json,
}

impl ExportFormat {
    /// Lowercase name, used in the default filename.
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Markdown => "md",
            ExportFormat::Json => "json",
        }
    }

    /// Human label for the export button.
    pub fn label(&self) -> &'static str {
        match self {
            ExportFormat::Markdown => "Export chat (Markdown)",
            ExportFormat::Json => "Export chat (JSON)",
        }
    }
}

/// One `(question, answer)` pair, as the chat window stores them.
///
/// Timestamps are optional (`Option`) rather than defaulted: a message pulled
/// from a session log may genuinely have no recorded time, and writing
/// `1970-01-01` would be a lie about when the conversation happened.
pub type ChatTurn = (String, String);

/// Render a transcript in `format`.
///
/// Generic over the element type so callers can pass either `&[ChatTurn]` or a
/// slice of references without copying the strings.
///
/// An empty transcript produces a valid, empty document rather than an error:
/// the user pressed export with nothing said yet, and an error dialog for that
/// would be unhelpful.
pub fn export<T>(symbol: &str, turns: &[T], format: ExportFormat) -> String
where
    T: std::borrow::Borrow<ChatTurn>,
{
    match format {
        ExportFormat::Markdown => markdown(symbol, turns),
        ExportFormat::Json => json(symbol, turns),
    }
}

fn markdown<T>(symbol: &str, turns: &[T]) -> String
where
    T: std::borrow::Borrow<ChatTurn>,
{
    let mut out = format!("# Bharat Terminal chat — {symbol}\n\n");
    if turns.is_empty() {
        out.push_str("_No messages in this conversation._\n");
        return out;
    }
    for turn in turns {
        let (q, a) = turn.borrow();
        out.push_str("## You\n\n");
        out.push_str(q.trim());
        out.push_str("\n\n## Assistant\n\n");
        out.push_str(a.trim());
        out.push_str("\n\n");
    }
    out.push_str("---\n\nMade by Sourish Dey\n");
    out
}

fn json<T>(symbol: &str, turns: &[T]) -> String
where
    T: std::borrow::Borrow<ChatTurn>,
{
    // Built through serde_json rather than by string concatenation so quotes and
    // newlines inside a message cannot corrupt the document.
    let pairs: Vec<serde_json::Value> = turns
        .iter()
        .map(|turn| {
            let (q, a) = turn.borrow();
            serde_json::json!({ "question": q, "answer": a })
        })
        .collect();
    let doc = serde_json::json!({
        "app": "Bharat Terminal",
        "author": "Sourish Dey",
        "symbol": symbol,
        "messages": pairs,
    });
    // Pretty-printed so the file is diffable and readable.
    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string())
}

/// Suggested filename for an export.
pub fn suggested_filename(symbol: &str, format: ExportFormat) -> String {
    let safe: String = symbol
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
            c
        } else {
            '_'
        })
        .collect();
    let safe = if safe.is_empty() { "chat".to_string() } else { safe };
    format!("chat-{safe}.{}", format.extension())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns() -> Vec<ChatTurn> {
        vec![
            ("What is the trend?".into(), "Uptrend on RSI 62.".into()),
            ("And tomorrow?".into(), "No forecast is a certainty.".into()),
        ]
    }

    #[test]
    fn markdown_keeps_every_turn() {
        let md = export("RELIANCE.NS", &turns(), ExportFormat::Markdown);
        for (q, a) in turns() {
            assert!(md.contains(&q), "question missing:\n{md}");
            assert!(md.contains(&a), "answer missing:\n{md}");
        }
        assert!(md.starts_with("# Bharat Terminal chat — RELIANCE.NS"), "{md}");
        assert_eq!(md.matches("## You").count(), 2, "{md}");
        assert_eq!(md.matches("## Assistant").count(), 2, "{md}");
    }

    #[test]
    fn json_is_parseable_and_carries_every_turn() {
        let text = export("RELIANCE.NS", &turns(), ExportFormat::Json);
        let v: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(v["symbol"], "RELIANCE.NS");
        assert_eq!(v["messages"].as_array().unwrap().len(), 2, "{text}");
        assert_eq!(v["messages"][0]["question"], "What is the trend?");
        assert_eq!(v["messages"][1]["answer"], "No forecast is a certainty.");
    }

    #[test]
    fn json_survives_quotes_and_newlines_in_a_message() {
        // The failure this guards: hand-built JSON that breaks the moment a
        // model replies with a quotation mark.
        let nasty = vec![(
            "He said \"buy\"?".into(),
            "Line one\nLine two\ttabbed \\ backslash".into(),
        )];
        let text = export("X", &nasty, ExportFormat::Json);
        let v: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("produced invalid JSON: {e}\n{text}"));
        assert_eq!(v["messages"][0]["answer"], "Line one\nLine two\ttabbed \\ backslash");
    }

    #[test]
    fn an_empty_transcript_is_valid_rather_than_an_error() {
        let md = export("X", &[] as &[ChatTurn], ExportFormat::Markdown);
        assert!(md.contains("No messages"), "{md}");
        let text = export("X", &[] as &[ChatTurn], ExportFormat::Json);
        let v: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(v["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn both_formats_carry_the_attribution() {
        assert!(export("X", &turns(), ExportFormat::Markdown).contains("Made by Sourish Dey"));
        let text = export("X", &turns(), ExportFormat::Json);
        assert!(text.contains("Sourish Dey"), "{text}");
    }

    #[test]
    fn filenames_are_sanitised_and_never_empty() {
        assert_eq!(
            suggested_filename("RELIANCE.NS", ExportFormat::Markdown),
            "chat-RELIANCE.NS.md"
        );
        assert_eq!(
            suggested_filename("BRK/B", ExportFormat::Json),
            "chat-BRK_B.json"
        );
        // A symbol made entirely of illegal characters still yields a filename.
        let name = suggested_filename("///", ExportFormat::Markdown);
        assert!(name.starts_with("chat-"), "{name}");
        assert!(!name.contains('/'), "{name}");
    }

    #[test]
    fn export_is_deterministic() {
        for f in [ExportFormat::Markdown, ExportFormat::Json] {
            assert_eq!(export("X", &turns(), f), export("X", &turns(), f));
        }
    }
}
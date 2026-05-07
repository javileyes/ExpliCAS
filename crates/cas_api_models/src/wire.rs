//! Wire model for serializable output.
//!
//! Provides a unified, stable output format for REPL/CLI/Web/FFI.
//! All consumers can use this same schema for consistent messaging.

use cas_ast::Span;
use serde::{Deserialize, Serialize};

use crate::{AssumptionDto, BlockedHintDto, WarningWire};

/// Current schema version for the wire format.
pub const SCHEMA_VERSION: u32 = 1;

/// Top-level wire response container.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WireReply {
    /// Schema version for forwards/backwards compatibility
    pub schema_version: u32,
    /// Messages in order of emission
    pub messages: Vec<WireMsg>,
}

impl WireReply {
    /// Create a new WireReply with current schema version.
    pub fn new(messages: Vec<WireMsg>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            messages,
        }
    }
}

/// Message kind for wire format.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WireKind {
    /// Main output/result.
    Output,
    /// Informational message.
    Info,
    /// Warning (non-fatal).
    Warn,
    /// Error (fatal).
    Error,
    /// Step-by-step trace.
    Steps,
    /// Debug output.
    Debug,
}

/// Individual message in wire format.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WireMsg {
    /// Message kind.
    pub kind: WireKind,
    /// Text content.
    pub text: String,
    /// Source span if available (for error localization).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<WireSpan>,
    /// Structured metadata for FFI/frontend (codes, rule names, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl WireMsg {
    /// Create a new wire message.
    pub fn new(kind: WireKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
            span: None,
            data: None,
        }
    }

    /// Create a new wire message with a span.
    pub fn with_span(kind: WireKind, text: impl Into<String>, span: Span) -> Self {
        Self {
            kind,
            text: text.into(),
            span: Some(span.into()),
            data: None,
        }
    }

    /// Create a new wire message with structured data.
    pub fn with_data(kind: WireKind, text: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            kind,
            text: text.into(),
            span: None,
            data: Some(data),
        }
    }

    /// Create a new wire message with span and structured data.
    pub fn with_span_and_data(
        kind: WireKind,
        text: impl Into<String>,
        span: Span,
        data: serde_json::Value,
    ) -> Self {
        Self {
            kind,
            text: text.into(),
            span: Some(span.into()),
            data: Some(data),
        }
    }

    /// Create a parse error with `E_PARSE` code and optional span.
    pub fn parse_error(message: impl Into<String>, span: Option<Span>) -> Self {
        use serde_json::json;
        let text = message.into();
        let data = json!({"code": "E_PARSE", "phase": "parse"});
        match span {
            Some(s) => Self {
                kind: WireKind::Error,
                text,
                span: Some(s.into()),
                data: Some(data),
            },
            None => Self {
                kind: WireKind::Error,
                text,
                span: None,
                data: Some(data),
            },
        }
    }

    /// Create a warning with stable code and optional originating rule.
    pub fn warning_with_code(code: &str, message: impl Into<String>, rule: Option<&str>) -> Self {
        use serde_json::json;
        let mut data = json!({"code": code});
        if let Some(r) = rule {
            data["rule"] = json!(r);
        }
        Self {
            kind: WireKind::Warn,
            text: message.into(),
            span: None,
            data: Some(data),
        }
    }

    /// Create an info message with stable code.
    pub fn info_with_code(code: &str, message: impl Into<String>) -> Self {
        use serde_json::json;
        Self {
            kind: WireKind::Info,
            text: message.into(),
            span: None,
            data: Some(json!({"code": code})),
        }
    }
}

/// Build a wire envelope for eval-style wire outputs.
pub struct EvalWireReplyParts<'a> {
    pub warnings: &'a [WarningWire],
    pub assumptions_used: &'a [AssumptionDto],
    pub required_display: &'a [String],
    pub blocked_hints: &'a [BlockedHintDto],
    pub strategy: Option<&'a str>,
    pub result: &'a str,
    pub result_latex: Option<&'a str>,
    pub steps_count: usize,
    pub steps_mode: &'a str,
}

/// Format blocked hints for human-facing wire/CLI messages.
///
/// The structured `blocked_hints` payload remains ungrouped. This helper only
/// reduces repeated condition/tip noise in display messages.
pub fn format_blocked_hint_message_lines(blocked_hints: &[BlockedHintDto]) -> Vec<String> {
    let mut grouped: Vec<(String, String, Vec<String>)> = Vec::new();

    for hint in blocked_hints {
        let requires = hint.requires.join(", ");
        if let Some((_, _, rules)) =
            grouped
                .iter_mut()
                .find(|(existing_requires, existing_tip, _)| {
                    existing_requires == &requires && existing_tip == &hint.tip
                })
        {
            if !rules.iter().any(|rule| rule == &hint.rule) {
                rules.push(hint.rule.clone());
            }
            continue;
        }

        grouped.push((requires, hint.tip.clone(), vec![hint.rule.clone()]));
    }

    let mut lines = Vec::new();
    for (requires, tip, rules) in grouped {
        lines.push(format!(
            "\u{2139}\u{FE0F} Blocked: requires {} [{}]",
            requires,
            rules.join(", ")
        ));
        lines.push(format!("  {tip}"));
    }

    lines
}

/// Build a wire envelope for eval-style wire outputs.
///
/// Message order:
/// 1. warnings
/// 2. assumptions used
/// 3. required conditions
/// 4. strategy (optional)
/// 5. result
/// 6. steps summary (optional)
pub fn build_eval_wire_reply(parts: EvalWireReplyParts<'_>) -> WireReply {
    let EvalWireReplyParts {
        warnings,
        assumptions_used,
        required_display,
        blocked_hints,
        strategy,
        result,
        result_latex,
        steps_count,
        steps_mode,
    } = parts;
    let mut messages = Vec::new();

    for w in warnings {
        messages.push(WireMsg::new(
            WireKind::Warn,
            format!("\u{26A0} {} ({})", w.assumption, w.rule),
        ));
    }

    if !assumptions_used.is_empty() {
        messages.push(WireMsg::new(WireKind::Info, "\u{2139}\u{FE0F} Assume:"));
        for assumption in assumptions_used {
            messages.push(WireMsg::new(
                WireKind::Info,
                format!("  \u{2022} {}", assumption.display),
            ));
        }
    }

    if !required_display.is_empty() {
        messages.push(WireMsg::new(WireKind::Info, "\u{2139}\u{FE0F} Requires:"));
        for cond in required_display {
            messages.push(WireMsg::new(WireKind::Info, format!("  \u{2022} {}", cond)));
        }
    }

    for line in format_blocked_hint_message_lines(blocked_hints) {
        messages.push(WireMsg::new(WireKind::Info, line));
    }

    if let Some(strategy) = strategy {
        messages.push(WireMsg::new(
            WireKind::Info,
            format!("Strategy: {strategy}"),
        ));
    }

    let result_text = if let Some(latex) = result_latex {
        format!("Result: {} [LaTeX: {}]", result, latex)
    } else {
        format!("Result: {}", result)
    };
    messages.push(WireMsg::new(WireKind::Output, result_text));

    if steps_mode == "on" && steps_count > 0 {
        let steps_text = if let Some(strategy) = strategy {
            format!("{steps_count} step(s) via {strategy}")
        } else {
            format!("{steps_count} simplification step(s)")
        };
        messages.push(WireMsg::new(WireKind::Steps, steps_text));
    }

    WireReply::new(messages)
}

/// Source span for wire format (matches `cas_ast::Span` but serializable).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireSpan {
    /// Start byte offset (inclusive).
    pub start: usize,
    /// End byte offset (exclusive).
    pub end: usize,
}

impl From<Span> for WireSpan {
    fn from(s: Span) -> Self {
        Self {
            start: s.start,
            end: s.end,
        }
    }
}

impl From<WireSpan> for Span {
    fn from(s: WireSpan) -> Self {
        Span::new(s.start, s.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_eval_wire_reply_adds_expected_sections() {
        let warnings = vec![WarningWire {
            rule: "r1".to_string(),
            assumption: "a != 0".to_string(),
        }];
        let required = vec!["x > 0".to_string()];
        let reply = build_eval_wire_reply(EvalWireReplyParts {
            warnings: &warnings,
            assumptions_used: &[],
            required_display: &required,
            blocked_hints: &[],
            strategy: Some("expand"),
            result: "42",
            result_latex: Some("42"),
            steps_count: 3,
            steps_mode: "on",
        });

        assert_eq!(reply.schema_version, SCHEMA_VERSION);
        assert_eq!(reply.messages.len(), 6);
        assert_eq!(reply.messages[0].kind, WireKind::Warn);
        assert_eq!(reply.messages[1].kind, WireKind::Info);
        assert_eq!(reply.messages[2].kind, WireKind::Info);
        assert_eq!(reply.messages[3].kind, WireKind::Info);
        assert_eq!(reply.messages[3].text, "Strategy: expand");
        assert_eq!(reply.messages[4].kind, WireKind::Output);
        assert_eq!(reply.messages[5].kind, WireKind::Steps);
        assert_eq!(reply.messages[5].text, "3 step(s) via expand");
    }

    #[test]
    fn build_eval_wire_reply_omits_steps_when_disabled() {
        let reply = build_eval_wire_reply(EvalWireReplyParts {
            warnings: &[],
            assumptions_used: &[],
            required_display: &[],
            blocked_hints: &[],
            strategy: None,
            result: "ok",
            result_latex: None,
            steps_count: 10,
            steps_mode: "off",
        });
        assert_eq!(reply.messages.len(), 1);
        assert_eq!(reply.messages[0].kind, WireKind::Output);
        assert_eq!(reply.messages[0].text, "Result: ok");
    }

    #[test]
    fn build_eval_wire_reply_falls_back_to_simplification_steps_without_strategy() {
        let reply = build_eval_wire_reply(EvalWireReplyParts {
            warnings: &[],
            assumptions_used: &[],
            required_display: &[],
            blocked_hints: &[],
            strategy: None,
            result: "ok",
            result_latex: None,
            steps_count: 2,
            steps_mode: "on",
        });
        assert_eq!(reply.messages.len(), 2);
        assert_eq!(reply.messages[1].kind, WireKind::Steps);
        assert_eq!(reply.messages[1].text, "2 simplification step(s)");
    }

    #[test]
    fn format_blocked_hint_message_lines_groups_repeated_condition_and_tip() {
        let hints = vec![
            BlockedHintDto {
                rule: "Cancel Identical Numerator/Denominator".to_string(),
                requires: vec!["x \u{2260} 0".to_string()],
                tip: "use `domain generic` to allow definability assumptions".to_string(),
            },
            BlockedHintDto {
                rule: "Simplify Nested Fraction".to_string(),
                requires: vec!["x \u{2260} 0".to_string()],
                tip: "use `domain generic` to allow definability assumptions".to_string(),
            },
            BlockedHintDto {
                rule: "Cancel Common Factors".to_string(),
                requires: vec!["x \u{2260} 0".to_string()],
                tip: "use `domain generic` to allow definability assumptions".to_string(),
            },
        ];

        let lines = format_blocked_hint_message_lines(&hints);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("x \u{2260} 0"));
        assert!(lines[0].contains("Cancel Identical Numerator/Denominator"));
        assert!(lines[0].contains("Simplify Nested Fraction"));
        assert!(lines[0].contains("Cancel Common Factors"));
        assert_eq!(
            lines[1],
            "  use `domain generic` to allow definability assumptions"
        );
    }
}

//! Wire model for serializable output.
//!
//! Provides a unified, stable output format for REPL/CLI/Web/FFI.
//! All consumers can use this same schema for consistent messaging.

use cas_ast::Span;
use serde::{Deserialize, Serialize};

use crate::WarningJson;

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

/// Build a wire envelope for eval-style JSON outputs.
///
/// Message order:
/// 1. warnings
/// 2. required conditions
/// 3. result
/// 4. steps summary (optional)
pub fn build_eval_wire_reply(
    warnings: &[WarningJson],
    required_display: &[String],
    result: &str,
    result_latex: Option<&str>,
    steps_count: usize,
    steps_mode: &str,
) -> WireReply {
    let mut messages = Vec::new();

    for w in warnings {
        messages.push(WireMsg::new(
            WireKind::Warn,
            format!("\u{26A0} {} ({})", w.assumption, w.rule),
        ));
    }

    if !required_display.is_empty() {
        messages.push(WireMsg::new(WireKind::Info, "\u{2139}\u{FE0F} Requires:"));
        for cond in required_display {
            messages.push(WireMsg::new(WireKind::Info, format!("  \u{2022} {}", cond)));
        }
    }

    let result_text = if let Some(latex) = result_latex {
        format!("Result: {} [LaTeX: {}]", result, latex)
    } else {
        format!("Result: {}", result)
    };
    messages.push(WireMsg::new(WireKind::Output, result_text));

    if steps_mode == "on" && steps_count > 0 {
        messages.push(WireMsg::new(
            WireKind::Steps,
            format!("{} simplification step(s)", steps_count),
        ));
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
        let warnings = vec![WarningJson {
            rule: "r1".to_string(),
            assumption: "a != 0".to_string(),
        }];
        let required = vec!["x > 0".to_string()];
        let reply = build_eval_wire_reply(&warnings, &required, "42", Some("42"), 3, "on");

        assert_eq!(reply.schema_version, SCHEMA_VERSION);
        assert_eq!(reply.messages.len(), 5);
        assert_eq!(reply.messages[0].kind, WireKind::Warn);
        assert_eq!(reply.messages[1].kind, WireKind::Info);
        assert_eq!(reply.messages[2].kind, WireKind::Info);
        assert_eq!(reply.messages[3].kind, WireKind::Output);
        assert_eq!(reply.messages[4].kind, WireKind::Steps);
    }

    #[test]
    fn build_eval_wire_reply_omits_steps_when_disabled() {
        let reply = build_eval_wire_reply(&[], &[], "ok", None, 10, "off");
        assert_eq!(reply.messages.len(), 1);
        assert_eq!(reply.messages[0].kind, WireKind::Output);
        assert_eq!(reply.messages[0].text, "Result: ok");
    }
}

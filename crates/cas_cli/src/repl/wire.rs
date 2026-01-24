//! Wire model for serializable output.
//!
//! Provides a unified, stable output format for REPL/CLI/Web/FFI.
//! All consumers can use this same schema for consistent messaging.
//!
//! # Stability Contract
//!
//! - `schema_version` allows evolution without breaking clients
//! - Field additions are backwards-compatible (use skip_serializing_if)

use cas_ast::Span;
use serde::{Deserialize, Serialize};

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
    /// Create a new WireReply with current schema version
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
    /// Main output/result
    Output,
    /// Informational message
    Info,
    /// Warning (non-fatal)
    Warn,
    /// Error (fatal)
    Error,
    /// Step-by-step trace
    Steps,
    /// Debug output
    Debug,
}

/// Individual message in wire format.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WireMsg {
    /// Message kind
    pub kind: WireKind,
    /// Text content
    pub text: String,
    /// Source span if available (for error localization)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<WireSpan>,
    /// Structured metadata for FFI/frontend (codes, rule names, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Source span for wire format (matches cas_ast::Span but serializable).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireSpan {
    /// Start byte offset (inclusive)
    pub start: usize,
    /// End byte offset (exclusive)
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

impl WireMsg {
    /// Create a new WireMsg
    pub fn new(kind: WireKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
            span: None,
            data: None,
        }
    }

    /// Create a new WireMsg with span
    pub fn with_span(kind: WireKind, text: impl Into<String>, span: Span) -> Self {
        Self {
            kind,
            text: text.into(),
            span: Some(span.into()),
            data: None,
        }
    }

    /// Create a new WireMsg with structured data
    pub fn with_data(kind: WireKind, text: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            kind,
            text: text.into(),
            span: None,
            data: Some(data),
        }
    }

    /// Create a new WireMsg with span and data
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
}

// =============================================================================
// Conversion from ReplMsg to WireMsg
// =============================================================================

use super::output::ReplMsg;

impl From<&ReplMsg> for WireMsg {
    fn from(msg: &ReplMsg) -> Self {
        match msg {
            ReplMsg::Output(s) => WireMsg::new(WireKind::Output, s),
            ReplMsg::Info(s) => WireMsg::new(WireKind::Info, s),
            ReplMsg::Warn(s) => WireMsg::new(WireKind::Warn, s),
            ReplMsg::Error(s) => WireMsg::new(WireKind::Error, s),
            ReplMsg::Steps(s) => WireMsg::new(WireKind::Steps, s),
            ReplMsg::Debug(s) => WireMsg::new(WireKind::Debug, s),
            // WriteFile is an action, not a displayable message — convert to Info
            ReplMsg::WriteFile { path, .. } => {
                WireMsg::new(WireKind::Info, format!("Wrote: {}", path.display()))
            }
        }
    }
}

impl From<ReplMsg> for WireMsg {
    fn from(msg: ReplMsg) -> Self {
        WireMsg::from(&msg)
    }
}

// =============================================================================
// Conversion from ReplReply to WireReply
// =============================================================================

use super::output::ReplReply;

impl From<&ReplReply> for WireReply {
    fn from(reply: &ReplReply) -> Self {
        let messages: Vec<WireMsg> = reply.iter().map(WireMsg::from).collect();
        WireReply::new(messages)
    }
}

impl From<ReplReply> for WireReply {
    fn from(reply: ReplReply) -> Self {
        WireReply::from(&reply)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_wire_msg_serialization() {
        let msg = WireMsg::new(WireKind::Output, "x = 5");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""kind":"output""#));
        assert!(json.contains(r#""text":"x = 5""#));
        // span should not be present when None
        assert!(!json.contains("span"));
    }

    #[test]
    fn test_wire_msg_with_span_serialization() {
        let msg = WireMsg::with_span(WireKind::Error, "bad token", Span::new(4, 7));
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""kind":"error""#));
        assert!(json.contains(r#""span""#));
        assert!(json.contains(r#""start":4"#));
        assert!(json.contains(r#""end":7"#));
    }

    #[test]
    fn test_wire_reply_serialization() {
        let reply = WireReply::new(vec![
            WireMsg::new(WireKind::Output, "result"),
            WireMsg::new(WireKind::Info, "note"),
        ]);
        let json = serde_json::to_string(&reply).unwrap();
        assert!(json.contains(r#""schema_version":1"#));
        assert!(json.contains(r#""messages""#));
    }

    #[test]
    fn test_round_trip_serialization() {
        let original = WireReply::new(vec![
            WireMsg::new(WireKind::Output, "x = 5"),
            WireMsg::with_span(WireKind::Error, "bad", Span::new(1, 3)),
        ]);
        let json = serde_json::to_string(&original).unwrap();
        let parsed: WireReply = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_repl_msg_to_wire_msg() {
        let msg = ReplMsg::Output("hello".to_string());
        let wire: WireMsg = (&msg).into();
        assert_eq!(wire.kind, WireKind::Output);
        assert_eq!(wire.text, "hello");
    }

    #[test]
    fn test_repl_reply_to_wire_reply() {
        let reply: ReplReply = vec![
            ReplMsg::Output("result".to_string()),
            ReplMsg::Warn("warning".to_string()),
        ];
        let wire: WireReply = (&reply).into();
        assert_eq!(wire.schema_version, SCHEMA_VERSION);
        assert_eq!(wire.messages.len(), 2);
        assert_eq!(wire.messages[0].kind, WireKind::Output);
        assert_eq!(wire.messages[1].kind, WireKind::Warn);
    }

    #[test]
    fn test_write_file_converts_to_info() {
        let msg = ReplMsg::WriteFile {
            path: PathBuf::from("/tmp/test.html"),
            contents: "data".to_string(),
        };
        let wire: WireMsg = (&msg).into();
        assert_eq!(wire.kind, WireKind::Info);
        assert!(wire.text.contains("Wrote:"));
    }

    #[test]
    fn test_wire_msg_with_data_serialization() {
        use serde_json::json;
        let msg = WireMsg::with_data(
            WireKind::Error,
            "parse error",
            json!({"code": "E_PARSE", "phase": "parse"}),
        );
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""kind":"error""#));
        assert!(json.contains(r#""code":"E_PARSE""#));
        assert!(json.contains(r#""phase":"parse""#));
    }

    #[test]
    fn test_wire_msg_with_span_and_data() {
        use serde_json::json;
        let msg = WireMsg::with_span_and_data(
            WireKind::Error,
            "unexpected token",
            Span::new(4, 5),
            json!({"code": "E_PARSE"}),
        );
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""span""#));
        assert!(json.contains(r#""start":4"#));
        assert!(json.contains(r#""code":"E_PARSE""#));
    }

    #[test]
    fn test_data_not_serialized_when_none() {
        let msg = WireMsg::new(WireKind::Output, "result");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("data"));
    }
}

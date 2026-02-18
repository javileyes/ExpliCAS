//! Wire model adapter for REPL output.
//!
//! Canonical wire DTOs live in `cas_api_models::wire`. This module keeps only
//! REPL-specific conversions.

pub use cas_api_models::wire::{WireKind, WireMsg, WireReply, WireSpan, SCHEMA_VERSION};

use super::output::{ReplMsg, ReplReply};

impl From<&ReplMsg> for WireMsg {
    fn from(msg: &ReplMsg) -> Self {
        match msg {
            ReplMsg::Output(s) => WireMsg::new(WireKind::Output, s),
            ReplMsg::Info(s) => WireMsg::new(WireKind::Info, s),
            ReplMsg::Warn(s) => WireMsg::new(WireKind::Warn, s),
            ReplMsg::Error(s) => WireMsg::new(WireKind::Error, s),
            ReplMsg::Steps(s) => WireMsg::new(WireKind::Steps, s),
            ReplMsg::Debug(s) => WireMsg::new(WireKind::Debug, s),
            // WriteFile is an action, not a displayable message — convert to Info.
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

/// Convert a REPL reply into the wire envelope.
pub fn wire_reply_from_repl(reply: &ReplReply) -> WireReply {
    let messages: Vec<WireMsg> = reply.iter().map(WireMsg::from).collect();
    WireReply::new(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Span;
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
        let wire = wire_reply_from_repl(&reply);
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

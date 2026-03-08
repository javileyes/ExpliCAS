use super::*;
use crate::repl::output::{ReplMsg, ReplReply};
use cas_ast::Span;
use std::path::PathBuf;

#[test]
fn test_wire_msg_serialization() {
    let msg = WireMsg::new(WireKind::Output, "x = 5");
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains(r#""kind":"output""#));
    assert!(json.contains(r#""text":"x = 5""#));
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
fn test_open_file_converts_to_info() {
    let msg = ReplMsg::OpenFile {
        path: PathBuf::from("/tmp/test.html"),
    };
    let wire: WireMsg = (&msg).into();
    assert_eq!(wire.kind, WireKind::Info);
    assert!(wire.text.contains("Open:"));
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

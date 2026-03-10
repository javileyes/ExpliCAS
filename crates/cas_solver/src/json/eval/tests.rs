use super::eval_str_to_json;

#[test]
fn eval_session_ref_returns_invalid_input() {
    let json = eval_str_to_json("#1 + x", "{}");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("json");

    assert_eq!(parsed["ok"], false);
    assert_eq!(parsed["error"]["kind"], "InvalidInput");
    assert_eq!(parsed["error"]["code"], "E_INVALID_INPUT");
}

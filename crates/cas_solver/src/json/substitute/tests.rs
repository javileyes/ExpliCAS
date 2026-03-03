use super::substitute_str_to_json;

#[test]
fn substitute_str_to_json_returns_ok_contract() {
    let out = substitute_str_to_json("x^2+1", "x", "y", Some(r#"{"mode":"exact"}"#));
    let json: serde_json::Value = serde_json::from_str(&out).expect("valid json");
    assert_eq!(json["ok"], true);
}

#[test]
fn substitute_str_to_json_parse_error_contract() {
    let out = substitute_str_to_json("x^2 + 1", "invalid(((", "y", None);
    let json: serde_json::Value = serde_json::from_str(&out).expect("valid json");
    assert_eq!(json["ok"], false);
    assert_eq!(json["error"]["kind"], "ParseError");
}

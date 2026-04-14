use super::eval_str_to_wire;

#[test]
fn eval_session_ref_returns_invalid_input() {
    let json = eval_str_to_wire("#1 + x", "{}");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("json");

    assert_eq!(parsed["ok"], false);
    assert_eq!(parsed["error"]["kind"], "InvalidInput");
    assert_eq!(parsed["error"]["code"], "E_INVALID_INPUT");
}

#[test]
fn eval_str_to_wire_handles_hyperbolic_passthrough_shifted_quotient_without_steps() {
    let json = eval_str_to_wire(
        "((2*sinh(2*x)*sinh(x)+a) + 1)/((4*cosh(x)^3-4*cosh(x)+a) + 1)",
        "{}",
    );
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("json");

    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["result"], "1");
}

use super::evaluate_envelope_json_command;

#[test]
fn evaluate_envelope_json_command_returns_json_contract() {
    let payload = evaluate_envelope_json_command("x + x", "generic", "real");
    assert!(payload.contains("\"schema_version\": 1"));
    assert!(payload.contains("\"kind\": \"eval_result\""));
}

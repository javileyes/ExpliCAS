use super::evaluate_envelope_wire_command;
use cas_api_models::{EvalDomainMode, EvalValueDomain};

#[test]
fn evaluate_envelope_wire_command_returns_wire_contract() {
    let payload =
        evaluate_envelope_wire_command("x + x", EvalDomainMode::Generic, EvalValueDomain::Real);
    assert!(payload.contains("\"schema_version\": 1"));
    assert!(payload.contains("\"kind\": \"eval_result\""));
}

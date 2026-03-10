#[cfg(test)]
mod tests {
    use crate::evaluate_envelope_wire_command;

    #[test]
    fn evaluate_envelope_wire_command_returns_json_contract() {
        let payload = evaluate_envelope_wire_command("x + x", "generic", "real");
        assert!(payload.contains("\"schema_version\": 1"));
        assert!(payload.contains("\"kind\": \"eval_result\""));
    }
}

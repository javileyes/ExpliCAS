//! Stateless CLI-subcommand helper for envelope-json.

/// Evaluate `envelope-json` command and return pretty JSON payload.
pub fn evaluate_envelope_json_command(expr: &str, domain: &str, value_domain: &str) -> String {
    let output = cas_solver::eval_str_to_output_envelope(
        expr,
        &cas_solver::EnvelopeEvalOptions {
            domain: domain.to_string(),
            value_domain: value_domain.to_string(),
        },
    );
    output.to_json_pretty()
}

#[cfg(test)]
mod tests {
    use super::evaluate_envelope_json_command;

    #[test]
    fn evaluate_envelope_json_command_returns_json_contract() {
        let payload = evaluate_envelope_json_command("x + x", "generic", "real");
        assert!(payload.contains("\"schema_version\": 1"));
        assert!(payload.contains("\"kind\": \"eval_result\""));
    }
}

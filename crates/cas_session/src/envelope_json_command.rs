//! Stateless CLI-subcommand helper for envelope-json.

use cas_api_models::EnvelopeEvalOptions;

use crate::envelope_json_command_runtime::eval_str_to_output_envelope;

/// Evaluate `envelope-json` command and return pretty JSON payload.
pub fn evaluate_envelope_json_command(expr: &str, domain: &str, value_domain: &str) -> String {
    let opts = EnvelopeEvalOptions {
        domain: domain.to_string(),
        value_domain: value_domain.to_string(),
    };
    let output = eval_str_to_output_envelope(expr, &opts);
    output.to_json_pretty()
}

use std::path::Path;

use super::{evaluate_eval_json_command_with_session, EvalJsonCommandConfig};
use cas_engine::Step;

/// Evaluate `eval-json` and always return a pretty JSON string.
///
/// Successful runs return canonical JSON payload. Errors are normalized into
/// canonical JSON error output.
pub fn evaluate_eval_json_command_pretty_with_session<F>(
    session_path: Option<&Path>,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> String
where
    F: Fn(&[Step], &cas_ast::Context, &str) -> Vec<cas_api_models::StepJson>,
{
    let input = config.expr.to_string();
    let (output, _, _) =
        evaluate_eval_json_command_with_session(session_path, config, collect_steps);
    match output {
        Ok(payload) => payload.to_json_pretty(),
        Err(error) => cas_api_models::ErrorJsonOutput::from_eval_error_message(&error, &input)
            .to_json_pretty(),
    }
}

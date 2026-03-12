use std::path::Path;

use super::{evaluate_eval_command_with_session, EvalCommandConfig};
use cas_engine::Step;
use cas_solver_core::engine_events::EngineEvent;

/// Evaluate `eval` and always return a pretty JSON string.
///
/// Successful runs return canonical JSON payload. Errors are normalized into
/// canonical JSON error output.
pub fn evaluate_eval_command_pretty_with_session<F>(
    session_path: Option<&Path>,
    config: EvalCommandConfig<'_>,
    collect_steps: F,
) -> String
where
    F: Fn(&[Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<cas_api_models::StepWire>,
{
    let input = config.expr;
    let (output, _, _) = evaluate_eval_command_with_session(session_path, config, collect_steps);
    match output {
        Ok(payload) => payload.to_json_pretty(),
        Err(error) => {
            cas_api_models::ErrorWireOutput::from_eval_error_message(&error, input).to_json_pretty()
        }
    }
}

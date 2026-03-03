//! Session-backed eval-json command orchestration.

use std::path::Path;

pub(crate) use crate::eval_json_command_runtime::evaluate_eval_json_with_session;

/// Session-backed config for `eval-json` command orchestration.
pub type EvalJsonCommandConfig<'a> = cas_api_models::EvalJsonSessionRunConfig<'a>;

/// Evaluate `eval-json` using optional persisted session state.
///
/// Keeps CLI/frontends thin by centralizing session load/run/save orchestration.
pub fn evaluate_eval_json_command_with_session<F>(
    session_path: Option<&Path>,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> (
    Result<cas_api_models::EvalJsonOutput, String>,
    Option<String>,
    Option<String>,
)
where
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<cas_api_models::StepJson>,
{
    crate::run_with_domain_session(session_path, config.domain, |engine, state| {
        evaluate_eval_json_with_session(engine, state, config, |steps, ctx, mode| {
            collect_steps(steps, ctx, mode)
        })
    })
}

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
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<cas_api_models::StepJson>,
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

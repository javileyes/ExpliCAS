use cas_engine::Step;
use std::path::Path;

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
    F: Fn(&[Step], &cas_ast::Context, &str) -> Vec<cas_api_models::StepJson>,
{
    crate::run_with_domain_session(session_path, config.domain, |engine, state| {
        cas_solver::evaluate_eval_json_with_session(engine, state, config, |steps, ctx, mode| {
            collect_steps(steps, ctx, mode)
        })
    })
}

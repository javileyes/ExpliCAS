use cas_engine::Step;
use cas_solver_core::engine_events::EngineEvent;
use std::path::Path;

/// Session-backed config for eval command orchestration.
pub type EvalCommandConfig<'a> = cas_api_models::EvalSessionRunConfig<'a>;
type EvalOutputWire = cas_api_models::EvalWireOutput;
type EvalCommandResult = Result<EvalOutputWire, String>;

/// Evaluate `eval` using optional persisted session state.
///
/// Keeps CLI/frontends thin by centralizing session load/run/save orchestration.
pub fn evaluate_eval_command_with_session<F>(
    session_path: Option<&Path>,
    config: EvalCommandConfig<'_>,
    collect_steps: F,
) -> (EvalCommandResult, Option<String>, Option<String>)
where
    F: Fn(&[Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<cas_api_models::StepWire>,
{
    crate::session_io::run_with_domain_session(session_path, config.domain, |engine, state| {
        cas_solver::evaluate_eval_with_session(engine, state, config, |steps, events, ctx, mode| {
            collect_steps(steps, events, ctx, mode)
        })
    })
}

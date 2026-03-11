use super::super::cli_actions::timeline_cli_actions_from_render;
use super::super::simplify::VerbosityLevel;
use super::super::types::TimelineCliAction;
use super::session;

pub fn extract_timeline_invocation_input(line: &str) -> &str {
    line.strip_prefix("timeline")
        .map(str::trim)
        .unwrap_or_else(|| line.trim())
}

pub fn evaluate_timeline_invocation_cli_actions_with_session<S>(
    engine: &mut crate::cas_solver::Engine,
    session: &mut S,
    line: &str,
    eval_options: &crate::cas_solver::EvalOptions,
    verbosity: VerbosityLevel,
) -> Result<Vec<TimelineCliAction>, cas_session::solver_exports::TimelineCommandEvalError>
where
    S: crate::cas_solver::EvalSession<
        Options = crate::cas_solver::EvalOptions,
        Diagnostics = crate::cas_solver::Diagnostics,
    >,
    S::Store: crate::cas_solver::EvalStore<
        DomainMode = crate::cas_solver::DomainMode,
        RequiredItem = crate::cas_solver::RequiredItem,
        Step = crate::cas_solver::Step,
        Diagnostics = crate::cas_solver::Diagnostics,
    >,
{
    let input = extract_timeline_invocation_input(line);
    let render = session::evaluate_timeline_command_cli_render_with_session(
        engine,
        session,
        input,
        eval_options,
        verbosity,
    )?;
    Ok(timeline_cli_actions_from_render(render))
}

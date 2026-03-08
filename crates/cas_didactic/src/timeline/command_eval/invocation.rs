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
    engine: &mut cas_solver::Engine,
    session: &mut S,
    line: &str,
    eval_options: &cas_solver::EvalOptions,
    verbosity: VerbosityLevel,
) -> Result<Vec<TimelineCliAction>, cas_session::TimelineCommandEvalError>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
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

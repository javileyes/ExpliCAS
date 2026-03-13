use super::super::cli_actions::timeline_cli_actions_from_render;
use super::super::simplify::VerbosityLevel;
use super::super::TimelineCliAction;
use super::session;
use cas_solver::session_api::timeline::TimelineCommandEvalError;

pub fn extract_timeline_invocation_input(line: &str) -> &str {
    line.strip_prefix("timeline")
        .map(str::trim)
        .unwrap_or_else(|| line.trim())
}

pub fn evaluate_timeline_invocation_cli_actions_with_session<S>(
    engine: &mut crate::runtime::Engine,
    session: &mut S,
    line: &str,
    eval_options: &crate::runtime::EvalOptions,
    verbosity: VerbosityLevel,
) -> Result<Vec<TimelineCliAction>, TimelineCommandEvalError>
where
    S: crate::runtime::EvalSession<
        Options = crate::runtime::EvalOptions,
        Diagnostics = crate::runtime::Diagnostics,
    >,
    S::Store: crate::runtime::EvalStore<
        DomainMode = crate::runtime::DomainMode,
        RequiredItem = crate::runtime::RequiredItem,
        Step = crate::runtime::Step,
        Diagnostics = crate::runtime::Diagnostics,
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

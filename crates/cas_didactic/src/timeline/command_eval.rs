use super::cli_actions::timeline_cli_actions_from_render;
use super::cli_output::render_timeline_command_cli_output;
use super::command_projection::timeline_command_output_from_solver;
use super::simplify::VerbosityLevel;
use super::types::{TimelineCliAction, TimelineCliRender, TimelineCommandOutput};

/// Evaluate a `timeline` command and project solver output to didactic payload.
pub fn evaluate_timeline_command_output_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
) -> Result<TimelineCommandOutput, cas_session::TimelineCommandEvalError>
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
    let output =
        cas_session::evaluate_timeline_command_with_session(engine, session, input, eval_options)?;
    Ok(timeline_command_output_from_solver(output))
}

/// Evaluate and render a `timeline` command to CLI render output in one call.
pub fn evaluate_timeline_command_cli_render_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
    verbosity: VerbosityLevel,
) -> Result<TimelineCliRender, cas_session::TimelineCommandEvalError>
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
    let out = evaluate_timeline_command_output_with_session(engine, session, input, eval_options)?;
    Ok(render_timeline_command_cli_output(
        &mut engine.simplifier.context,
        &out,
        verbosity,
    ))
}

/// Extract timeline input from a full invocation or return trimmed input as-is.
pub fn extract_timeline_invocation_input(line: &str) -> &str {
    line.strip_prefix("timeline")
        .map(str::trim)
        .unwrap_or_else(|| line.trim())
}

/// Evaluate a full `timeline ...` invocation and return normalized CLI actions.
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
    let render = evaluate_timeline_command_cli_render_with_session(
        engine,
        session,
        input,
        eval_options,
        verbosity,
    )?;
    Ok(timeline_cli_actions_from_render(render))
}

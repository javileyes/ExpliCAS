use super::super::super::cli_output::render_timeline_command_cli_output;
use super::super::super::simplify::VerbosityLevel;
use super::super::super::types::TimelineCliRender;
use super::eval::evaluate_timeline_command_output_with_session;

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
    let output =
        evaluate_timeline_command_output_with_session(engine, session, input, eval_options)?;
    Ok(render_timeline_command_cli_output(
        &mut engine.simplifier.context,
        &output,
        verbosity,
    ))
}

use super::super::super::cli_output::render_timeline_command_cli_output;
use super::super::super::simplify::VerbosityLevel;
use super::super::super::types::TimelineCliRender;
use super::eval::evaluate_timeline_command_output_with_session;

pub fn evaluate_timeline_command_cli_render_with_session<S>(
    engine: &mut crate::cas_solver::Engine,
    session: &mut S,
    input: &str,
    eval_options: &crate::cas_solver::EvalOptions,
    verbosity: VerbosityLevel,
) -> Result<TimelineCliRender, cas_session::solver_exports::TimelineCommandEvalError>
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
    let output =
        evaluate_timeline_command_output_with_session(engine, session, input, eval_options)?;
    Ok(render_timeline_command_cli_output(
        &mut engine.simplifier.context,
        &output,
        verbosity,
    ))
}

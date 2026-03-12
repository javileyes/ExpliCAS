use super::super::super::cli_output::render_timeline_command_cli_output;
use super::super::super::simplify::VerbosityLevel;
use super::super::super::types::TimelineCliRender;
use super::eval::evaluate_timeline_command_output_with_session;
use cas_solver::session_api::timeline::TimelineCommandEvalError;

pub fn evaluate_timeline_command_cli_render_with_session<S>(
    engine: &mut crate::runtime::Engine,
    session: &mut S,
    input: &str,
    eval_options: &crate::runtime::EvalOptions,
    verbosity: VerbosityLevel,
) -> Result<TimelineCliRender, TimelineCommandEvalError>
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
    let output =
        evaluate_timeline_command_output_with_session(engine, session, input, eval_options)?;
    Ok(render_timeline_command_cli_output(
        &mut engine.simplifier.context,
        &output,
        verbosity,
    ))
}

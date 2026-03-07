use super::super::super::command_projection::timeline_command_output_from_solver;
use super::super::super::types::TimelineCommandOutput;

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

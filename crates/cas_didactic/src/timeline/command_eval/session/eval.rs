use super::super::super::command_projection::timeline_command_output_from_solver;
use super::super::super::types::TimelineCommandOutput;

pub fn evaluate_timeline_command_output_with_session<S>(
    engine: &mut crate::cas_solver::Engine,
    session: &mut S,
    input: &str,
    eval_options: &crate::cas_solver::EvalOptions,
) -> Result<TimelineCommandOutput, cas_session::solver_exports::TimelineCommandEvalError>
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
    let output = cas_session::solver_exports::evaluate_timeline_command_with_session(
        engine,
        session,
        input,
        eval_options,
    )?;
    Ok(timeline_command_output_from_solver(output))
}

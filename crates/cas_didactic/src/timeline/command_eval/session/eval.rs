use super::super::super::command_projection::timeline_command_output_from_solver;
use super::super::super::types::TimelineCommandOutput;
use cas_solver::session_api::timeline::{
    evaluate_timeline_command_with_session, TimelineCommandEvalError,
};

pub fn evaluate_timeline_command_output_with_session<S>(
    engine: &mut crate::runtime::Engine,
    session: &mut S,
    input: &str,
    eval_options: &crate::runtime::EvalOptions,
) -> Result<TimelineCommandOutput, TimelineCommandEvalError>
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
    let output = evaluate_timeline_command_with_session(engine, session, input, eval_options)?;
    Ok(timeline_command_output_from_solver(output))
}

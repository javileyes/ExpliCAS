mod execute;
mod output;
mod types;

use self::types::SolveSessionExecution;
use super::{SolveCommandEvalError, SolveCommandEvalOutput};

pub(crate) fn evaluate_solve_parsed_with_session<S>(
    simplifier: &mut crate::Simplifier,
    session: &mut S,
    raw_input: String,
    parsed_expr: cas_ast::ExprId,
    var: &str,
    auto_store: bool,
) -> Result<crate::EvalOutputView, String>
where
    S: crate::SolverEvalSession,
{
    let execution = execute::solve_parsed_with_session(
        simplifier,
        session,
        raw_input,
        parsed_expr,
        var,
        auto_store,
    )?;
    Ok(output::finalize_solve_eval_output(
        session, simplifier, execution,
    ))
}

pub fn evaluate_solve_command_with_session<S>(
    simplifier: &mut crate::Simplifier,
    session: &mut S,
    parsed_input: crate::SolveCommandInput,
    auto_store: bool,
) -> Result<SolveCommandEvalOutput, SolveCommandEvalError>
where
    S: crate::SolverEvalSession,
{
    let prepared = super::prepare_solve_eval_request(
        &mut simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
        auto_store,
    )
    .map_err(SolveCommandEvalError::Prepare)?;

    let output = evaluate_solve_parsed_with_session(
        simplifier,
        session,
        prepared.raw_input.clone(),
        prepared.parsed_expr,
        &prepared.var,
        prepared.auto_store,
    )
    .map_err(SolveCommandEvalError::Eval)?;

    Ok(SolveCommandEvalOutput {
        var: prepared.var,
        original_equation: prepared.original_equation,
        output,
    })
}

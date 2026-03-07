use crate::eval_command_types::{EvalCommandError, EvalCommandEvalView};

pub(super) fn execute_eval_request<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    req: crate::EvalRequest,
) -> Result<EvalCommandEvalView, EvalCommandError>
where
    S: crate::SolverEvalSession,
{
    let output = engine
        .eval(session, req)
        .map_err(|e| EvalCommandError::Eval(format!("Error: {}", e)))?;
    let output_view = crate::eval_output_view(&output);
    Ok(EvalCommandEvalView {
        stored_id: output_view.stored_id,
        parsed: output_view.parsed,
        resolved: output_view.resolved,
        result: output_view.result,
        diagnostics: output_view.diagnostics,
        steps: output_view.steps,
        domain_warnings: output_view.domain_warnings,
        blocked_hints: output_view.blocked_hints,
    })
}

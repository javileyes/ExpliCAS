use crate::eval_command_format_result::format_eval_result_text;

/// Evaluate plain text simplification input and return final rendered result.
///
/// This is a thin solver-level orchestration helper for CLI/frontends that
/// want `parse -> eval(simplify) -> render result` with a stateful session.
pub fn evaluate_eval_text_simplify_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    expr: &str,
    auto_store: bool,
) -> Result<String, String>
where
    S: crate::SolverEvalSession,
{
    let req = crate::eval_input::build_prepared_eval_request_for_input(
        expr,
        &mut engine.simplifier.context,
        auto_store,
    )?;
    let output_view =
        crate::eval_request_runtime::evaluate_prepared_request_with_session(engine, session, req)
            .map_err(|e| {
            if e.starts_with("Error:") || e.starts_with("Parse error:") || e.starts_with("Usage:") {
                e
            } else {
                format!("Error: {e}")
            }
        })?;
    Ok(format_eval_result_text(
        &engine.simplifier.context,
        &output_view.result,
    ))
}

use crate::eval_command_format::format_eval_result_text;

/// Evaluate plain text simplification input and return final rendered result.
///
/// This is a thin solver-level orchestration helper for CLI/frontends that
/// want `parse -> eval(simplify) -> render result` with a stateful session.
pub fn evaluate_eval_text_simplify_with_session(
    engine: &mut cas_solver::Engine,
    session: &mut crate::SessionState,
    expr: &str,
    auto_store: bool,
) -> Result<String, String> {
    let parsed = cas_parser::parse(expr, &mut engine.simplifier.context)
        .map_err(|e| format!("Parse error: {}", e))?;
    let req = cas_solver::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: cas_solver::EvalAction::Simplify,
        auto_store,
    };
    let output = engine
        .eval(session, req)
        .map_err(|e| format!("Error: {}", e))?;
    let output_view = cas_solver::eval_output_view(&output);
    Ok(format_eval_result_text(
        &engine.simplifier.context,
        &output_view.result,
    ))
}

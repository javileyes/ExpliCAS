use crate::eval_command_format::format_eval_result_text;

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
    let parsed = cas_parser::parse(expr, &mut engine.simplifier.context)
        .map_err(|e| format!("Parse error: {}", e))?;
    let req = crate::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: crate::EvalAction::Simplify,
        auto_store,
    };
    let output = engine
        .eval(session, req)
        .map_err(|e| format!("Error: {}", e))?;
    let output_view = crate::eval_output_view(&output);
    Ok(format_eval_result_text(
        &engine.simplifier.context,
        &output_view.result,
    ))
}

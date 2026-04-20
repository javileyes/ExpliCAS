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
    let previous_session_steps_mode = session.options().steps_mode;
    let previous_simplifier_steps_mode = engine.simplifier.get_steps_mode();
    let previous_listener = engine.simplifier.replace_step_listener(None);

    if !matches!(previous_session_steps_mode, crate::StepsMode::Off) {
        session.options_mut().steps_mode = crate::StepsMode::Off;
    }
    engine.simplifier.set_steps_mode(crate::StepsMode::Off);

    let output_view_result =
        crate::eval_request_runtime::evaluate_prepared_request_with_session(engine, session, req);

    if session.options().steps_mode != previous_session_steps_mode {
        session.options_mut().steps_mode = previous_session_steps_mode;
    }
    engine
        .simplifier
        .set_steps_mode(previous_simplifier_steps_mode);
    engine.simplifier.set_step_listener(previous_listener);

    let output_view = output_view_result.map_err(|e| {
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

use crate::full_simplify_render_steps::format_full_simplify_eval_lines;

/// Extract expression tail from a `simplify` command line.
pub fn extract_simplify_command_tail(line: &str) -> &str {
    line.strip_prefix("simplify").unwrap_or(line).trim()
}

/// Evaluate a full `simplify ...` invocation and return final display lines.
pub fn evaluate_full_simplify_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    session: &crate::SessionState,
    line: &str,
    display_mode: crate::SetDisplayMode,
) -> Result<Vec<String>, String> {
    let expr_input = extract_simplify_command_tail(line);
    let output = crate::evaluate_full_simplify_input(
        simplifier,
        session,
        expr_input,
        !matches!(display_mode, crate::SetDisplayMode::None),
    )
    .map_err(|error| crate::format_full_simplify_eval_error_message(&error))?;

    Ok(format_full_simplify_eval_lines(
        &mut simplifier.context,
        expr_input,
        &output,
        display_mode,
    ))
}

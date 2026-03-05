/// Evaluate and format unary command output lines.
pub fn evaluate_unary_function_command_lines(
    simplifier: &mut crate::Simplifier,
    function_name: &str,
    input: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
) -> Result<Vec<String>, String> {
    let parsed_expr = cas_parser::parse(input.trim(), &mut simplifier.context)
        .map_err(|e| format!("Parse error: {e}"))?;
    let call_expr = simplifier.context.call(function_name, vec![parsed_expr]);
    let (result_expr, steps) = simplifier.simplify(call_expr);
    Ok(crate::format_unary_function_eval_lines(
        &simplifier.context,
        input,
        result_expr,
        &steps,
        function_name,
        display_mode != crate::SetDisplayMode::None,
        show_step_assumptions,
    ))
}

/// Evaluate unary command line (`det ...`, `trace ...`, etc.) and optionally
/// normalize final `Result:` display line.
pub fn evaluate_unary_command_lines(
    simplifier: &mut crate::Simplifier,
    line: &str,
    command: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
    clean_result_line: bool,
) -> Result<Vec<String>, String> {
    let rest = line.strip_prefix(command).unwrap_or(line).trim();
    let mut lines = evaluate_unary_function_command_lines(
        simplifier,
        command,
        rest,
        display_mode,
        show_step_assumptions,
    )?;
    if clean_result_line {
        crate::clean_result_output_line(&mut lines);
    }
    Ok(lines)
}

/// Evaluate unary command line and return final message text.
pub fn evaluate_unary_command_message(
    simplifier: &mut crate::Simplifier,
    line: &str,
    command: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
    clean_result_line: bool,
) -> Result<String, String> {
    Ok(evaluate_unary_command_lines(
        simplifier,
        line,
        command,
        display_mode,
        show_step_assumptions,
        clean_result_line,
    )?
    .join("\n"))
}

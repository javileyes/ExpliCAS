pub(super) fn evaluate_unary_command_lines(
    simplifier: &mut crate::Simplifier,
    line: &str,
    command: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
    clean_result_line: bool,
) -> Result<Vec<String>, String> {
    let rest = line.strip_prefix(command).unwrap_or(line).trim();
    let mut lines = super::function_eval::evaluate_unary_function_command_lines(
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

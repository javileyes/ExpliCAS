pub(super) fn evaluate_unary_command_message(
    simplifier: &mut crate::Simplifier,
    line: &str,
    command: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
    clean_result_line: bool,
) -> Result<String, String> {
    Ok(super::lines::evaluate_unary_command_lines(
        simplifier,
        line,
        command,
        display_mode,
        show_step_assumptions,
        clean_result_line,
    )?
    .join("\n"))
}

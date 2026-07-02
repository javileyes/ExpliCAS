mod function_eval;
mod lines;
mod message;

/// Evaluate unary command line and return final message text.
pub(crate) fn evaluate_unary_command_message(
    simplifier: &mut crate::Simplifier,
    line: &str,
    command: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
    clean_result_line: bool,
) -> Result<String, String> {
    message::evaluate_unary_command_message(
        simplifier,
        line,
        command,
        display_mode,
        show_step_assumptions,
        clean_result_line,
    )
}

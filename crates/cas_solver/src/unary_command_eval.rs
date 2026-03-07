mod function_eval;
mod lines;
mod message;

/// Evaluate and format unary command output lines.
pub fn evaluate_unary_function_command_lines(
    simplifier: &mut crate::Simplifier,
    function_name: &str,
    input: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
) -> Result<Vec<String>, String> {
    function_eval::evaluate_unary_function_command_lines(
        simplifier,
        function_name,
        input,
        display_mode,
        show_step_assumptions,
    )
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
    lines::evaluate_unary_command_lines(
        simplifier,
        line,
        command,
        display_mode,
        show_step_assumptions,
        clean_result_line,
    )
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
    message::evaluate_unary_command_message(
        simplifier,
        line,
        command,
        display_mode,
        show_step_assumptions,
        clean_result_line,
    )
}

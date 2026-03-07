use crate::substitute_command_eval::eval::evaluate_substitute_command_lines;
use crate::substitute_command_format::{
    format_substitute_parse_error_message, substitute_render_mode_from_display_mode,
};
use crate::substitute_command_types::SubstituteParseError;

/// Evaluate full `subst ...` invocation line and return cleaned display lines.
pub fn evaluate_substitute_invocation_lines(
    simplifier: &mut crate::Simplifier,
    line: &str,
    display_mode: crate::SetDisplayMode,
) -> Result<Vec<String>, SubstituteParseError> {
    let input = crate::extract_substitute_command_tail(line);
    let mode = substitute_render_mode_from_display_mode(display_mode);
    let mut lines = evaluate_substitute_command_lines(simplifier, input, mode)?;
    crate::clean_result_output_line(&mut lines);
    Ok(lines)
}

/// Evaluate full `subst ...` invocation line and return cleaned message text.
pub fn evaluate_substitute_invocation_message(
    simplifier: &mut crate::Simplifier,
    line: &str,
    display_mode: crate::SetDisplayMode,
) -> Result<String, SubstituteParseError> {
    Ok(evaluate_substitute_invocation_lines(simplifier, line, display_mode)?.join("\n"))
}

/// Evaluate full `subst ...` invocation and return user-facing text message.
pub fn evaluate_substitute_invocation_user_message(
    simplifier: &mut crate::Simplifier,
    line: &str,
    display_mode: crate::SetDisplayMode,
) -> Result<String, String> {
    evaluate_substitute_invocation_message(simplifier, line, display_mode)
        .map_err(|error| format_substitute_parse_error_message(&error))
}

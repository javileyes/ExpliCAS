//! CLI-level helpers for `limit` command parsing/formatting.

mod error;
mod lines;
mod parse;

/// Evaluate `limit` command input and return final display lines.
pub fn evaluate_limit_command_lines(line: &str) -> Result<Vec<String>, String> {
    evaluate_limit_command_lines_in_domain(line, false)
}

/// Domain-aware variant: threads the session's value domain into the limit
/// engine so the F0 kill-switch applies on the REPL surface too (a complex
/// session must decline every limit honestly, never run the real-order rules).
pub fn evaluate_limit_command_lines_in_domain(
    line: &str,
    complex_enabled: bool,
) -> Result<Vec<String>, String> {
    let rest = parse::extract_limit_command_tail(line);
    if rest.is_empty() {
        return Err(error::LIMIT_USAGE_MESSAGE.to_string());
    }

    let output =
        crate::limit_command_core::evaluate_limit_command_input_in_domain(rest, complex_enabled)
            .map_err(|error| error::format_limit_command_error_message(&error))?;
    Ok(lines::format_limit_command_eval_lines(&output))
}

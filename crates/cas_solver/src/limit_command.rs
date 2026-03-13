//! CLI-level helpers for `limit` command parsing/formatting.

mod error;
mod lines;
mod parse;

use crate::limit_command_core::evaluate_limit_command_input;

/// Evaluate `limit` command input and return final display lines.
pub fn evaluate_limit_command_lines(line: &str) -> Result<Vec<String>, String> {
    let rest = parse::extract_limit_command_tail(line);
    if rest.is_empty() {
        return Err(error::LIMIT_USAGE_MESSAGE.to_string());
    }

    let output = evaluate_limit_command_input(rest)
        .map_err(|error| error::format_limit_command_error_message(&error))?;
    Ok(lines::format_limit_command_eval_lines(&output))
}

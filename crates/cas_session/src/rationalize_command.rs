//! Session-level helpers for `rationalize` command parsing/formatting.

use crate::rationalize_command_eval::evaluate_rationalize_command_input;
use crate::rationalize_command_format::format_rationalize_eval_lines;
use crate::rationalize_command_parse::parse_rationalize_input;
use crate::rationalize_command_types::{RationalizeCommandEvalError, RATIONALIZE_USAGE_MESSAGE};

/// Evaluate `rationalize` command and return final display lines.
pub fn evaluate_rationalize_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
) -> Result<Vec<String>, String> {
    let Some(rest) = parse_rationalize_input(line) else {
        return Err(RATIONALIZE_USAGE_MESSAGE.to_string());
    };

    let output =
        evaluate_rationalize_command_input(simplifier, rest).map_err(|error| match error {
            RationalizeCommandEvalError::Parse(message) => format!("Parse error: {}", message),
        })?;

    Ok(format_rationalize_eval_lines(
        &simplifier.context,
        output.normalized_expr,
        output.outcome,
    ))
}

use cas_ast::Context;

use crate::ExplainCommandEvalError;

use super::gcd::evaluate_explain_command_lines;

/// Evaluate `explain` command input and return cleaned message text.
pub fn evaluate_explain_command_message(
    ctx: &mut Context,
    input: &str,
) -> Result<String, ExplainCommandEvalError> {
    let mut lines = evaluate_explain_command_lines(ctx, input)?;
    crate::clean_result_output_line(&mut lines);
    Ok(lines.join("\n"))
}

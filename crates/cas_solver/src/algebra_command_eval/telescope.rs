use crate::algebra_command_parse::{parse_telescope_invocation_input, telescope_usage_message};

/// Evaluate and format `telescope` command output lines.
pub fn evaluate_telescope_command_lines(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<Vec<String>, String> {
    let parsed_expr =
        cas_parser::parse(input.trim(), ctx).map_err(|e| format!("Parse error: {e}"))?;
    let result = crate::telescope(ctx, parsed_expr);
    let formatted_result = result.format(ctx);
    Ok(vec![format!("Parsed: {input}\n\n{formatted_result}")])
}

/// Evaluate `telescope ...` invocation and return display lines.
pub fn evaluate_telescope_invocation_lines(
    ctx: &mut cas_ast::Context,
    line: &str,
) -> Result<Vec<String>, String> {
    let Some(rest) = parse_telescope_invocation_input(line) else {
        return Err(telescope_usage_message().to_string());
    };
    evaluate_telescope_command_lines(ctx, &rest)
}

/// Evaluate `telescope ...` invocation and return display text.
pub fn evaluate_telescope_invocation_message(
    ctx: &mut cas_ast::Context,
    line: &str,
) -> Result<String, String> {
    Ok(evaluate_telescope_invocation_lines(ctx, line)?.join("\n"))
}

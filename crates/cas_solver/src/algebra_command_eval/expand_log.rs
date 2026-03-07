use crate::algebra_command_parse::{expand_log_usage_message, parse_expand_log_invocation_input};

/// Evaluate and format `expand_log` command output lines.
pub fn evaluate_expand_log_command_lines(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<Vec<String>, String> {
    let parsed_expr =
        cas_parser::parse(input.trim(), ctx).map_err(|e| format!("Parse error: {e}"))?;
    let expanded_expr = crate::expand_log_recursive(ctx, parsed_expr);
    Ok(vec![
        format!(
            "Parsed: {}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: parsed_expr
            }
        ),
        format!(
            "Result: {}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: expanded_expr
            }
        ),
    ])
}

/// Evaluate `expand_log ...` invocation and return display lines.
pub fn evaluate_expand_log_invocation_lines(
    ctx: &mut cas_ast::Context,
    line: &str,
) -> Result<Vec<String>, String> {
    let Some(rest) = parse_expand_log_invocation_input(line) else {
        return Err(expand_log_usage_message().to_string());
    };
    evaluate_expand_log_command_lines(ctx, &rest)
}

/// Evaluate `expand_log ...` invocation and return cleaned display text.
pub fn evaluate_expand_log_invocation_message(
    ctx: &mut cas_ast::Context,
    line: &str,
) -> Result<String, String> {
    let mut lines = evaluate_expand_log_invocation_lines(ctx, line)?;
    crate::clean_result_output_line(&mut lines);
    Ok(lines.join("\n"))
}

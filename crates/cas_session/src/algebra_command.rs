//! Session-level helpers for algebra REPL commands.

const TELESCOPE_USAGE_MESSAGE: &str = "Usage: telescope <expression>\n\
                 Example: telescope 1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)";
const EXPAND_USAGE_MESSAGE: &str = "Usage: expand <expr>\n\
                 Description: Aggressively expands and distributes polynomials.\n\
                 Example: expand 1/2 * (sqrt(2) - 1) -> sqrt(2)/2 - 1/2";
const EXPAND_LOG_USAGE_MESSAGE: &str = "Usage: expand_log <expr>\n\
                 Description: Expand logarithms using log properties.\n\
                 Transformations:\n\
                   ln(x*y)   -> ln(x) + ln(y)\n\
                   ln(x/y)   -> ln(x) - ln(y)\n\
                   ln(x^n)   -> n * ln(x)\n\
                 Example: expand_log ln(x^2 * y) -> 2*ln(x) + ln(y)";

/// Usage string for `telescope`.
pub fn telescope_usage_message() -> &'static str {
    TELESCOPE_USAGE_MESSAGE
}

/// Usage string for `expand`.
pub fn expand_usage_message() -> &'static str {
    EXPAND_USAGE_MESSAGE
}

/// Usage string for `expand_log`.
pub fn expand_log_usage_message() -> &'static str {
    EXPAND_LOG_USAGE_MESSAGE
}

/// Parse `telescope ...` invocation and return input expression.
pub fn parse_telescope_invocation_input(line: &str) -> Option<String> {
    let rest = line.strip_prefix("telescope").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Parse `expand ...` invocation and return input expression.
pub fn parse_expand_invocation_input(line: &str) -> Option<String> {
    let rest = line.strip_prefix("expand").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Parse `expand_log ...` invocation and return input expression.
pub fn parse_expand_log_invocation_input(line: &str) -> Option<String> {
    let rest = line.strip_prefix("expand_log").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

/// Wrap expression as explicit `expand(...)` call.
pub fn wrap_expand_eval_expression(expr: &str) -> String {
    format!("expand({expr})")
}

/// Parse and wrap `expand ...` as an explicit `expand(...)` eval input.
pub fn evaluate_expand_wrapped_expression(line: &str) -> Result<String, String> {
    let Some(rest) = parse_expand_invocation_input(line) else {
        return Err(expand_usage_message().to_string());
    };
    Ok(wrap_expand_eval_expression(&rest))
}

/// Evaluate and format `telescope` command output lines.
pub fn evaluate_telescope_command_lines(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<Vec<String>, String> {
    let eval_output = cas_solver::evaluate_telescope_command(ctx, input)?;
    let formatted_result = eval_output.result.format(ctx);
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

/// Evaluate and format `expand_log` command output lines.
pub fn evaluate_expand_log_command_lines(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<Vec<String>, String> {
    let eval_output = cas_solver::evaluate_expand_log_command(ctx, input)?;
    Ok(vec![
        format!(
            "Parsed: {}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: eval_output.parsed_expr
            }
        ),
        format!(
            "Result: {}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: eval_output.expanded_expr
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

#[cfg(test)]
mod tests {
    #[test]
    fn parse_telescope_invocation_input_reads_tail() {
        assert_eq!(
            super::parse_telescope_invocation_input("telescope 1+cos(x)"),
            Some("1+cos(x)".to_string())
        );
    }

    #[test]
    fn parse_expand_invocation_input_reads_tail() {
        assert_eq!(
            super::parse_expand_invocation_input("expand (x+1)^2"),
            Some("(x+1)^2".to_string())
        );
    }

    #[test]
    fn parse_expand_log_invocation_input_reads_tail() {
        assert_eq!(
            super::parse_expand_log_invocation_input("expand_log ln(x*y)"),
            Some("ln(x*y)".to_string())
        );
    }

    #[test]
    fn evaluate_telescope_command_lines_runs() {
        let mut ctx = cas_ast::Context::new();
        let lines =
            super::evaluate_telescope_command_lines(&mut ctx, "1 + 2*cos(x)").expect("telescope");
        assert!(lines.iter().any(|line| line.contains("Parsed:")));
    }

    #[test]
    fn evaluate_expand_log_command_lines_runs() {
        let mut ctx = cas_ast::Context::new();
        let lines =
            super::evaluate_expand_log_command_lines(&mut ctx, "ln(x^2*y)").expect("expand_log");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_expand_wrapped_expression_requires_input() {
        let err = super::evaluate_expand_wrapped_expression("expand").expect_err("usage");
        assert!(err.contains("Usage: expand"));
    }

    #[test]
    fn evaluate_telescope_invocation_lines_requires_input() {
        let mut ctx = cas_ast::Context::new();
        let err =
            super::evaluate_telescope_invocation_lines(&mut ctx, "telescope").expect_err("usage");
        assert!(err.contains("Usage: telescope"));
    }

    #[test]
    fn evaluate_expand_log_invocation_lines_requires_input() {
        let mut ctx = cas_ast::Context::new();
        let err =
            super::evaluate_expand_log_invocation_lines(&mut ctx, "expand_log").expect_err("usage");
        assert!(err.contains("Usage: expand_log"));
    }

    #[test]
    fn evaluate_telescope_invocation_message_joins_lines() {
        let mut ctx = cas_ast::Context::new();
        let message = super::evaluate_telescope_invocation_message(&mut ctx, "telescope 1+cos(x)")
            .expect("telescope");
        assert!(message.contains("Parsed: 1+cos(x)"));
    }

    #[test]
    fn evaluate_expand_log_invocation_message_returns_result_line() {
        let mut ctx = cas_ast::Context::new();
        let message = super::evaluate_expand_log_invocation_message(&mut ctx, "expand_log ln(x*y)")
            .expect("expand_log");
        assert!(message.contains("Result:"));
    }
}

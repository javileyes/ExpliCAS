const WEIERSTRASS_USAGE_MESSAGE: &str = "Usage: weierstrass <expression>\n\
                 Description: Apply Weierstrass substitution (t = tan(x/2))\n\
                 Transforms:\n\
                   sin(x) -> 2t/(1+t^2)\n\
                   cos(x) -> (1-t^2)/(1+t^2)\n\
                   tan(x) -> 2t/(1-t^2)\n\
                 Example: weierstrass sin(x) + cos(x)";

/// Usage string for `weierstrass`.
pub fn weierstrass_usage_message() -> &'static str {
    WEIERSTRASS_USAGE_MESSAGE
}

/// Parse `weierstrass ...` invocation and return expression tail.
pub fn parse_weierstrass_invocation_input(line: &str) -> Option<String> {
    let rest = line.strip_prefix("weierstrass").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

fn format_weierstrass_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    substituted_expr: cas_ast::ExprId,
    simplified_expr: cas_ast::ExprId,
) -> Vec<String> {
    vec![
        format!("Parsed: {input}"),
        String::new(),
        "Weierstrass substitution (t = tan(x/2)):".to_string(),
        format!(
            "  {} → {}",
            input,
            cas_formatter::DisplayExpr {
                context,
                id: substituted_expr
            }
        ),
        String::new(),
        "Simplifying...".to_string(),
        format!(
            "Result: {}",
            cas_formatter::DisplayExpr {
                context,
                id: simplified_expr
            }
        ),
    ]
}

/// Evaluate and format `weierstrass` command output lines.
pub fn evaluate_weierstrass_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<Vec<String>, String> {
    let parsed_expr = cas_parser::parse(input.trim(), &mut simplifier.context)
        .map_err(|e| format!("Parse error: {e}"))?;
    let substituted_expr =
        cas_solver::apply_weierstrass_recursive(&mut simplifier.context, parsed_expr);
    let (simplified_expr, _steps) = simplifier.simplify(substituted_expr);
    Ok(format_weierstrass_eval_lines(
        &simplifier.context,
        input,
        substituted_expr,
        simplified_expr,
    ))
}

/// Evaluate and format `weierstrass ...` invocation.
pub fn evaluate_weierstrass_invocation_lines(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
) -> Result<Vec<String>, String> {
    let Some(rest) = parse_weierstrass_invocation_input(line) else {
        return Err(weierstrass_usage_message().to_string());
    };
    evaluate_weierstrass_command_lines(simplifier, &rest)
}

/// Evaluate and format `weierstrass ...` invocation as message text.
pub fn evaluate_weierstrass_invocation_message(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
) -> Result<String, String> {
    let mut lines = evaluate_weierstrass_invocation_lines(simplifier, line)?;
    crate::clean_result_output_line(&mut lines);
    Ok(lines.join("\n"))
}

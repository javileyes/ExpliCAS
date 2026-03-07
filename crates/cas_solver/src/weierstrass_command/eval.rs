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
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<Vec<String>, String> {
    let parsed_expr = cas_parser::parse(input.trim(), &mut simplifier.context)
        .map_err(|e| format!("Parse error: {e}"))?;
    let substituted_expr = crate::apply_weierstrass_recursive(&mut simplifier.context, parsed_expr);
    let (simplified_expr, _steps) = simplifier.simplify(substituted_expr);
    Ok(format_weierstrass_eval_lines(
        &simplifier.context,
        input,
        substituted_expr,
        simplified_expr,
    ))
}

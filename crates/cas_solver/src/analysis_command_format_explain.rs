/// Format explain-gcd output as multi-line CLI text lines.
pub fn format_explain_gcd_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    steps: &[String],
    value: Option<cas_ast::ExprId>,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("Parsed: {input}"));
    lines.push(String::new());
    lines.push("Educational Steps:".to_string());
    lines.push("─".repeat(60));

    for step in steps {
        lines.push(step.clone());
    }

    lines.push("─".repeat(60));
    lines.push(String::new());

    if let Some(result_expr) = value {
        lines.push(format!(
            "Result: {}",
            cas_formatter::DisplayExpr {
                context,
                id: result_expr
            }
        ));
    } else {
        lines.push("Could not compute GCD".to_string());
    }
    lines
}

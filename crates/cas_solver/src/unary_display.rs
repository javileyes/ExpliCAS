/// Format unary command output (`det`, `trace`, etc.) with optional steps.
pub fn format_unary_function_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    result_expr: cas_ast::ExprId,
    steps: &[crate::Step],
    function_name: &str,
    show_steps: bool,
    show_step_assumptions: bool,
) -> Vec<String> {
    let mut lines = vec![format!("Parsed: {}({})", function_name, input)];

    if show_steps && !steps.is_empty() {
        lines.push("Steps:".to_string());
        for (i, step) in steps.iter().enumerate() {
            lines.push(format!(
                "{}. {}  [{}]",
                i + 1,
                step.description,
                step.rule_name
            ));
            if show_step_assumptions {
                for assumption_line in
                    crate::assumption_format::format_displayable_assumption_lines_for_step(step)
                {
                    lines.push(format!("   {assumption_line}"));
                }
            }
        }
    }

    lines.push(format!(
        "Result: {}",
        cas_formatter::DisplayExpr {
            context,
            id: result_expr
        }
    ));
    lines
}

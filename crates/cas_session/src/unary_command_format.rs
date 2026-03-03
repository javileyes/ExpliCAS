pub(crate) fn format_unary_function_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    result_expr: cas_ast::ExprId,
    steps: &[cas_solver::Step],
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
                let assumption_events = cas_solver::assumption_events_from_step(step);
                for assumption_line in
                    crate::format_displayable_assumption_lines(&assumption_events)
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

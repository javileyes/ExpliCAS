//! Session-level orchestration and formatting for unary algebra commands.

/// Evaluate and format unary command output lines.
pub fn evaluate_unary_function_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    function_name: &str,
    input: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
) -> Result<Vec<String>, String> {
    let eval_output =
        cas_solver::evaluate_unary_function_command(simplifier, function_name, input)?;
    Ok(format_unary_function_eval_lines(
        &simplifier.context,
        input,
        eval_output.result_expr,
        &eval_output.steps,
        function_name,
        display_mode != crate::SetDisplayMode::None,
        show_step_assumptions,
    ))
}

/// Evaluate unary command line (`det ...`, `trace ...`, etc.) and optionally
/// normalize final `Result:` display line.
pub fn evaluate_unary_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
    command: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
    clean_result_line: bool,
) -> Result<Vec<String>, String> {
    let rest = line.strip_prefix(command).unwrap_or(line).trim();
    let mut lines = evaluate_unary_function_command_lines(
        simplifier,
        command,
        rest,
        display_mode,
        show_step_assumptions,
    )?;
    if clean_result_line {
        crate::clean_result_output_line(&mut lines);
    }
    Ok(lines)
}

/// Evaluate unary command line and return final message text.
pub fn evaluate_unary_command_message(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
    command: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
    clean_result_line: bool,
) -> Result<String, String> {
    Ok(evaluate_unary_command_lines(
        simplifier,
        line,
        command,
        display_mode,
        show_step_assumptions,
        clean_result_line,
    )?
    .join("\n"))
}

fn format_unary_function_eval_lines(
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

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_unary_function_command_lines_runs() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let lines = super::evaluate_unary_function_command_lines(
            &mut simplifier,
            "trace",
            "[[1,2],[3,4]]",
            crate::SetDisplayMode::Normal,
            false,
        )
        .expect("unary eval");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_unary_command_lines_trims_command_prefix() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let lines = super::evaluate_unary_command_lines(
            &mut simplifier,
            "trace [[1,2],[3,4]]",
            "trace",
            crate::SetDisplayMode::None,
            false,
            true,
        )
        .expect("unary eval");
        assert!(lines
            .first()
            .is_some_and(|line| line.starts_with("Parsed: trace(")));
    }

    #[test]
    fn evaluate_unary_command_message_joins_lines() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let message = super::evaluate_unary_command_message(
            &mut simplifier,
            "trace [[1,2],[3,4]]",
            "trace",
            crate::SetDisplayMode::None,
            false,
            true,
        )
        .expect("unary eval");
        assert!(message.contains("Result:"));
    }
}

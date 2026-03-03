//! Evaluation helpers for analysis commands (`equiv`, `visualize`, `explain`).

/// Session-level output for `visualize` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisualizeCommandOutput {
    pub file_name: String,
    pub dot_source: String,
    pub hint_lines: Vec<String>,
}

fn visualize_output_hint_lines(file_name: &str) -> Vec<String> {
    vec![
        format!("Render with: dot -Tsvg {file_name} -o ast.svg"),
        format!("Or: dot -Tpng {file_name} -o ast.png"),
    ]
}

/// Evaluate equivalence command input and format user-facing output lines.
pub fn evaluate_equiv_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<Vec<String>, cas_solver::ParseExprPairError> {
    let result = cas_solver::evaluate_equiv_input(simplifier, input)?;
    Ok(crate::format_equivalence_result_lines(&result))
}

/// Evaluate `equiv` command input and return user-facing message text.
pub fn evaluate_equiv_command_message(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<String, cas_solver::ParseExprPairError> {
    Ok(evaluate_equiv_command_lines(simplifier, input)?.join("\n"))
}

/// Evaluate full `equiv ...` invocation and return user-facing message text.
pub fn evaluate_equiv_invocation_message(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
) -> Result<String, String> {
    let input = crate::extract_equiv_command_tail(line);
    evaluate_equiv_command_message(simplifier, input)
        .map_err(|error| crate::format_expr_pair_parse_error_message(&error, "equiv"))
}

/// Evaluate visualize command input and return DOT graph source.
pub fn evaluate_visualize_command_dot(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<String, cas_solver::VisualizeEvalError> {
    cas_solver::evaluate_visualize_ast_dot(ctx, input)
}

/// Evaluate visualize command input and return session-level output payload.
pub fn evaluate_visualize_command_output(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<VisualizeCommandOutput, cas_solver::VisualizeEvalError> {
    let file_name = "ast.dot";
    let dot_source = evaluate_visualize_command_dot(ctx, input)?;
    Ok(VisualizeCommandOutput {
        file_name: file_name.to_string(),
        dot_source,
        hint_lines: visualize_output_hint_lines(file_name),
    })
}

/// Evaluate full `visualize ...` invocation and return session-level output payload.
pub fn evaluate_visualize_invocation_output(
    ctx: &mut cas_ast::Context,
    line: &str,
) -> Result<VisualizeCommandOutput, String> {
    let input = crate::extract_visualize_command_tail(line);
    evaluate_visualize_command_output(ctx, input)
        .map_err(|error| crate::format_visualize_command_error_message(&error))
}

/// Evaluate explain command input and format user-facing output lines.
pub fn evaluate_explain_command_lines(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<Vec<String>, cas_solver::ExplainCommandEvalError> {
    let result = cas_solver::evaluate_explain_gcd_command(ctx, input)?;
    Ok(crate::format_explain_gcd_eval_lines(
        ctx,
        input,
        &result.steps,
        result.value,
    ))
}

/// Evaluate `explain` command input and return cleaned message text.
pub fn evaluate_explain_command_message(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<String, cas_solver::ExplainCommandEvalError> {
    let mut lines = evaluate_explain_command_lines(ctx, input)?;
    crate::clean_result_output_line(&mut lines);
    Ok(lines.join("\n"))
}

/// Evaluate full `explain ...` invocation and return user-facing message text.
pub fn evaluate_explain_invocation_message(
    ctx: &mut cas_ast::Context,
    line: &str,
) -> Result<String, String> {
    let input = crate::extract_explain_command_tail(line);
    evaluate_explain_command_message(ctx, input)
        .map_err(|error| crate::format_explain_command_error_message(&error))
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_equiv_command_lines_true() {
        let mut simplifier = cas_solver::Simplifier::new();
        let lines = super::evaluate_equiv_command_lines(&mut simplifier, "x+1, 1+x")
            .expect("equiv should evaluate");
        assert!(lines.iter().any(|line| line.contains("True")));
    }

    #[test]
    fn evaluate_visualize_command_dot_parse_error() {
        let mut ctx = cas_ast::Context::new();
        let err =
            super::evaluate_visualize_command_dot(&mut ctx, "x+").expect_err("expected parse");
        assert!(matches!(err, cas_solver::VisualizeEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_visualize_command_output_sets_file_and_hints() {
        let mut ctx = cas_ast::Context::new();
        let out = super::evaluate_visualize_command_output(&mut ctx, "x+1")
            .expect("visualize should evaluate");
        assert_eq!(out.file_name, "ast.dot");
        assert!(out.dot_source.contains("digraph"));
        assert_eq!(out.hint_lines.len(), 2);
        assert!(out.hint_lines[0].contains("dot -Tsvg"));
    }

    #[test]
    fn evaluate_explain_command_lines_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let lines = super::evaluate_explain_command_lines(&mut ctx, "gcd(8, 6)")
            .expect("explain should evaluate");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_equiv_command_message_joins_lines() {
        let mut simplifier = cas_solver::Simplifier::new();
        let message = super::evaluate_equiv_command_message(&mut simplifier, "x+1,1+x")
            .expect("equiv should evaluate");
        assert!(message.contains("True"));
    }

    #[test]
    fn evaluate_equiv_invocation_message_formats_parse_error() {
        let mut simplifier = cas_solver::Simplifier::new();
        let message = super::evaluate_equiv_invocation_message(&mut simplifier, "equiv x+1")
            .expect_err("parse");
        assert!(message.contains("equiv"));
    }

    #[test]
    fn evaluate_explain_command_message_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let message = super::evaluate_explain_command_message(&mut ctx, "gcd(8, 6)")
            .expect("explain should evaluate");
        assert!(message.contains("Result:"));
    }

    #[test]
    fn evaluate_explain_invocation_message_contains_result() {
        let mut ctx = cas_ast::Context::new();
        let message = super::evaluate_explain_invocation_message(&mut ctx, "explain gcd(8, 6)")
            .expect("explain should evaluate");
        assert!(message.contains("Result:"));
    }

    #[test]
    fn evaluate_visualize_invocation_output_sets_file() {
        let mut ctx = cas_ast::Context::new();
        let out = super::evaluate_visualize_invocation_output(&mut ctx, "visualize x+1")
            .expect("visualize should evaluate");
        assert_eq!(out.file_name, "ast.dot");
    }
}

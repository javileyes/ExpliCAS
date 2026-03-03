//! Formatting and invocation helpers for solve-related REPL commands.

use cas_ast::Context;

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

/// Format solve-prepare errors for end-user display.
pub fn format_solve_prepare_error_message(error: &cas_solver::SolvePrepareError) -> String {
    match error {
        cas_solver::SolvePrepareError::ParseError(e) => format!("Parse error: {e}"),
        cas_solver::SolvePrepareError::NoVariable => {
            "Error: solve() found no variable to solve for.\n\
                     Use solve(expr, x) to specify the variable."
                .to_string()
        }
        cas_solver::SolvePrepareError::AmbiguousVariables(vars) => format!(
            "Error: solve() found ambiguous variables {{{}}}.\n\
                     Use solve(expr, {}) or solve(expr, {{{}}}).",
            vars.join(", "),
            vars.first().unwrap_or(&"x".to_string()),
            vars.join(", ")
        ),
        cas_solver::SolvePrepareError::ExpectedEquation => {
            "Parse error: expected equation".to_string()
        }
    }
}

/// Format solve command evaluation errors for end-user display.
pub fn format_solve_command_error_message(error: &cas_solver::SolveCommandEvalError) -> String {
    match error {
        cas_solver::SolveCommandEvalError::Prepare(prepare) => {
            format_solve_prepare_error_message(prepare)
        }
        cas_solver::SolveCommandEvalError::Eval(e) => format!("Error: {e}"),
    }
}

/// Format verification summary lines after solve result checking.
pub fn format_verify_summary_lines(
    ctx: &Context,
    var: &str,
    verify_result: &cas_solver::VerifyResult,
    detail_prefix: &str,
) -> Vec<String> {
    let mut lines = Vec::new();

    match verify_result.summary {
        cas_solver::VerifySummary::AllVerified => {
            lines.push("✓ All solutions verified".to_string());
        }
        cas_solver::VerifySummary::PartiallyVerified => {
            lines.push("⚠ Some solutions verified".to_string());
            for (sol_id, status) in &verify_result.solutions {
                let sol_str = cas_formatter::render_expr(ctx, *sol_id);
                match status {
                    cas_solver::VerifyStatus::Verified => {
                        lines.push(format!("{detail_prefix}✓ {var} = {sol_str} verified"));
                    }
                    cas_solver::VerifyStatus::Unverifiable { reason, .. } => {
                        lines.push(format!("{detail_prefix}⚠ {var} = {sol_str}: {reason}"));
                    }
                    cas_solver::VerifyStatus::NotCheckable { reason } => {
                        lines.push(format!("{detail_prefix}ℹ {var} = {sol_str}: {reason}"));
                    }
                }
            }
        }
        cas_solver::VerifySummary::NoneVerified => {
            lines.push("⚠ No solutions could be verified".to_string());
        }
        cas_solver::VerifySummary::NotCheckable => {
            if let Some(desc) = &verify_result.guard_description {
                lines.push(format!("ℹ {desc}"));
            } else {
                lines.push("ℹ Solution type not checkable".to_string());
            }
        }
        cas_solver::VerifySummary::Empty => {}
    }

    lines
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

#[cfg(test)]
mod tests {
    #[test]
    fn parse_weierstrass_invocation_input_reads_tail() {
        assert_eq!(
            super::parse_weierstrass_invocation_input("weierstrass sin(x)+cos(x)"),
            Some("sin(x)+cos(x)".to_string())
        );
    }

    #[test]
    fn format_solve_command_error_message_prepare() {
        let msg = super::format_solve_command_error_message(
            &cas_solver::SolveCommandEvalError::Prepare(cas_solver::SolvePrepareError::NoVariable),
        );
        assert!(msg.contains("no variable"));
    }

    #[test]
    fn evaluate_weierstrass_command_lines_runs() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let lines = super::evaluate_weierstrass_command_lines(&mut simplifier, "sin(x)+cos(x)")
            .expect("weierstrass eval");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_weierstrass_invocation_lines_requires_input() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let err = super::evaluate_weierstrass_invocation_lines(&mut simplifier, "weierstrass")
            .expect_err("usage");
        assert!(err.contains("Usage: weierstrass"));
    }

    #[test]
    fn evaluate_weierstrass_invocation_message_joins_lines() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let message =
            super::evaluate_weierstrass_invocation_message(&mut simplifier, "weierstrass sin(x)")
                .expect("weierstrass");
        assert!(message.contains("Result:"));
    }
}

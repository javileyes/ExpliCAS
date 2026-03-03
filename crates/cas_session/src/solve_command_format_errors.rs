use cas_ast::Context;

/// Format solve-prepare errors for end-user display.
pub fn format_solve_prepare_error_message(error: &crate::SolvePrepareError) -> String {
    match error {
        crate::SolvePrepareError::ParseError(e) => format!("Parse error: {e}"),
        crate::SolvePrepareError::NoVariable => "Error: solve() found no variable to solve for.\n\
                     Use solve(expr, x) to specify the variable."
            .to_string(),
        crate::SolvePrepareError::AmbiguousVariables(vars) => format!(
            "Error: solve() found ambiguous variables {{{}}}.\n\
                     Use solve(expr, {}) or solve(expr, {{{}}}).",
            vars.join(", "),
            vars.first().unwrap_or(&"x".to_string()),
            vars.join(", ")
        ),
        crate::SolvePrepareError::ExpectedEquation => "Parse error: expected equation".to_string(),
    }
}

/// Format solve command evaluation errors for end-user display.
pub fn format_solve_command_error_message(error: &crate::SolveCommandEvalError) -> String {
    match error {
        crate::SolveCommandEvalError::Prepare(prepare) => {
            format_solve_prepare_error_message(prepare)
        }
        crate::SolveCommandEvalError::Eval(e) => format!("Error: {e}"),
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

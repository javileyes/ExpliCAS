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

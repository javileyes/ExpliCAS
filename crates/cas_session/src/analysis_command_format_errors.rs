/// Format timeline command error into user-facing message.
pub fn format_timeline_command_error_message(error: &crate::TimelineCommandEvalError) -> String {
    match error {
        crate::TimelineCommandEvalError::Solve(e) => format_timeline_solve_error_message(e),
        crate::TimelineCommandEvalError::Simplify(e) => format_timeline_eval_error_message(e),
    }
}

fn format_timeline_solve_error_message(error: &crate::TimelineSolveEvalError) -> String {
    match error {
        crate::TimelineSolveEvalError::Prepare(crate::SolvePrepareError::ExpectedEquation) => {
            "Error: Expected an equation for solve timeline, got an expression.\n\
                     Usage: timeline solve <equation>, <variable>\n\
                     Example: timeline solve x + 2 = 5, x"
                .to_string()
        }
        crate::TimelineSolveEvalError::Prepare(crate::SolvePrepareError::ParseError(e)) => {
            format!("Error parsing equation: {e}")
        }
        crate::TimelineSolveEvalError::Prepare(crate::SolvePrepareError::NoVariable) => {
            "Error: timeline solve found no variable.\n\
                 Use timeline solve <equation>, <variable>"
                .to_string()
        }
        crate::TimelineSolveEvalError::Prepare(crate::SolvePrepareError::AmbiguousVariables(
            vars,
        )) => {
            format!(
                "Error: timeline solve found ambiguous variables {{{}}}.\n\
                 Use timeline solve <equation>, {}",
                vars.join(", "),
                vars.first().unwrap_or(&"x".to_string())
            )
        }
        crate::TimelineSolveEvalError::Solve(e) => format!("Error solving: {e}"),
    }
}

fn format_timeline_eval_error_message(error: &crate::TimelineSimplifyEvalError) -> String {
    match error {
        crate::TimelineSimplifyEvalError::Parse(e) => format!("Parse error: {e}"),
        crate::TimelineSimplifyEvalError::Eval(e) => format!("Simplification error: {e}"),
    }
}

/// Format explain command errors into user-facing messages.
pub fn format_explain_command_error_message(error: &crate::ExplainCommandEvalError) -> String {
    match error {
        crate::ExplainCommandEvalError::Parse(e) => format!("Parse error: {e}"),
        crate::ExplainCommandEvalError::ExpectedFunctionCall => {
            "Explain mode currently only supports function calls\n\
             Try: explain gcd(48, 18)"
                .to_string()
        }
        crate::ExplainCommandEvalError::UnsupportedFunction(name) => format!(
            "Explain mode not yet implemented for function '{}'\n\
             Currently supported: gcd",
            name
        ),
        crate::ExplainCommandEvalError::InvalidArity {
            function, expected, ..
        } => {
            if function == "gcd" && *expected == 2 {
                "Usage: explain gcd(a, b)".to_string()
            } else {
                format!("Invalid arity for '{function}'")
            }
        }
    }
}

/// Format visualize command errors into user-facing messages.
pub fn format_visualize_command_error_message(error: &crate::VisualizeEvalError) -> String {
    match error {
        crate::VisualizeEvalError::Parse(message) => format!("Parse error: {message}"),
    }
}

//! Shared formatting for analysis-style REPL commands.

/// Format equivalence check result as display lines.
pub fn format_equivalence_result_lines(result: &cas_solver::EquivalenceResult) -> Vec<String> {
    match result {
        cas_solver::EquivalenceResult::True => vec!["True".to_string()],
        cas_solver::EquivalenceResult::ConditionalTrue { requires } => {
            let mut lines = vec!["True (conditional)".to_string()];
            lines.extend(format_text_requires_lines(requires));
            lines
        }
        cas_solver::EquivalenceResult::False => vec!["False".to_string()],
        cas_solver::EquivalenceResult::Unknown => {
            vec!["Unknown (cannot prove equivalence)".to_string()]
        }
    }
}

/// Format parse errors for commands that expect `<expr1>, <expr2>` input.
pub fn format_expr_pair_parse_error_message(
    error: &cas_solver::ParseExprPairError,
    command: &str,
) -> String {
    match error {
        cas_solver::ParseExprPairError::MissingDelimiter => {
            format!("Usage: {command} <expr1>, <expr2>")
        }
        cas_solver::ParseExprPairError::FirstArg(e) => format!("Error parsing first arg: {e}"),
        cas_solver::ParseExprPairError::SecondArg(e) => {
            format!("Error parsing second arg: {e}")
        }
    }
}

fn format_text_requires_lines(requires: &[String]) -> Vec<String> {
    if requires.is_empty() {
        return Vec::new();
    }

    let mut lines = vec!["ℹ️ Requires:".to_string()];
    for req in requires {
        lines.push(format!("  • {req}"));
    }
    lines
}

/// Format timeline command error into user-facing message.
pub fn format_timeline_command_error_message(
    error: &cas_solver::TimelineCommandEvalError,
) -> String {
    match error {
        cas_solver::TimelineCommandEvalError::Solve(e) => format_timeline_solve_error_message(e),
        cas_solver::TimelineCommandEvalError::Simplify(e) => format_timeline_eval_error_message(e),
    }
}

fn format_timeline_solve_error_message(error: &cas_solver::TimelineSolveEvalError) -> String {
    match error {
        cas_solver::TimelineSolveEvalError::Prepare(
            cas_solver::SolvePrepareError::ExpectedEquation,
        ) => "Error: Expected an equation for solve timeline, got an expression.\n\
                     Usage: timeline solve <equation>, <variable>\n\
                     Example: timeline solve x + 2 = 5, x"
            .to_string(),
        cas_solver::TimelineSolveEvalError::Prepare(cas_solver::SolvePrepareError::ParseError(
            e,
        )) => {
            format!("Error parsing equation: {e}")
        }
        cas_solver::TimelineSolveEvalError::Prepare(cas_solver::SolvePrepareError::NoVariable) => {
            "Error: timeline solve found no variable.\n\
                 Use timeline solve <equation>, <variable>"
                .to_string()
        }
        cas_solver::TimelineSolveEvalError::Prepare(
            cas_solver::SolvePrepareError::AmbiguousVariables(vars),
        ) => {
            format!(
                "Error: timeline solve found ambiguous variables {{{}}}.\n\
                 Use timeline solve <equation>, {}",
                vars.join(", "),
                vars.first().unwrap_or(&"x".to_string())
            )
        }
        cas_solver::TimelineSolveEvalError::Solve(e) => format!("Error solving: {e}"),
    }
}

fn format_timeline_eval_error_message(error: &cas_solver::TimelineSimplifyEvalError) -> String {
    match error {
        cas_solver::TimelineSimplifyEvalError::Parse(e) => format!("Parse error: {e}"),
        cas_solver::TimelineSimplifyEvalError::Eval(e) => format!("Simplification error: {e}"),
    }
}

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

/// Format explain command errors into user-facing messages.
pub fn format_explain_command_error_message(error: &cas_solver::ExplainCommandEvalError) -> String {
    match error {
        cas_solver::ExplainCommandEvalError::Parse(e) => format!("Parse error: {e}"),
        cas_solver::ExplainCommandEvalError::ExpectedFunctionCall => {
            "Explain mode currently only supports function calls\n\
             Try: explain gcd(48, 18)"
                .to_string()
        }
        cas_solver::ExplainCommandEvalError::UnsupportedFunction(name) => format!(
            "Explain mode not yet implemented for function '{}'\n\
             Currently supported: gcd",
            name
        ),
        cas_solver::ExplainCommandEvalError::InvalidArity {
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
pub fn format_visualize_command_error_message(error: &cas_solver::VisualizeEvalError) -> String {
    match error {
        cas_solver::VisualizeEvalError::Parse(message) => format!("Parse error: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        format_equivalence_result_lines, format_explain_command_error_message,
        format_expr_pair_parse_error_message, format_timeline_command_error_message,
        format_visualize_command_error_message,
    };

    #[test]
    fn format_equivalence_result_lines_conditional_includes_requires() {
        let lines =
            format_equivalence_result_lines(&cas_solver::EquivalenceResult::ConditionalTrue {
                requires: vec!["x != 0".to_string()],
            });
        assert!(lines.iter().any(|line| line.contains("conditional")));
        assert!(lines.iter().any(|line| line.contains("x != 0")));
    }

    #[test]
    fn format_timeline_command_error_message_parse_is_human_readable() {
        let err = cas_solver::TimelineCommandEvalError::Simplify(
            cas_solver::TimelineSimplifyEvalError::Parse("bad input".to_string()),
        );
        let msg = format_timeline_command_error_message(&err);
        assert!(msg.contains("Parse error"));
    }

    #[test]
    fn format_explain_command_error_message_parse_is_human_readable() {
        let msg = format_explain_command_error_message(
            &cas_solver::ExplainCommandEvalError::Parse("oops".to_string()),
        );
        assert!(msg.contains("Parse error"));
    }

    #[test]
    fn format_visualize_command_error_message_parse_is_human_readable() {
        let msg = format_visualize_command_error_message(&cas_solver::VisualizeEvalError::Parse(
            "bad".to_string(),
        ));
        assert!(msg.contains("Parse error"));
    }

    #[test]
    fn format_expr_pair_parse_error_message_usage_is_human_readable() {
        let msg = format_expr_pair_parse_error_message(
            &cas_solver::ParseExprPairError::MissingDelimiter,
            "equiv",
        );
        assert!(msg.contains("Usage: equiv"));
    }
}

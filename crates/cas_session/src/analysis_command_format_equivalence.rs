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
    error: &crate::ParseExprPairError,
    command: &str,
) -> String {
    match error {
        crate::ParseExprPairError::MissingDelimiter => {
            format!("Usage: {command} <expr1>, <expr2>")
        }
        crate::ParseExprPairError::FirstArg(e) => format!("Error parsing first arg: {e}"),
        crate::ParseExprPairError::SecondArg(e) => {
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

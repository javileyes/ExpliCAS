/// Format equivalence check result as display lines.
pub(crate) fn format_equivalence_result_lines(result: &crate::EquivalenceResult) -> Vec<String> {
    match result {
        crate::EquivalenceResult::True => vec!["True".to_string()],
        crate::EquivalenceResult::ConditionalTrue { requires } => {
            let mut lines = vec!["True (conditional)".to_string()];
            lines.extend(format_text_requires_lines(requires));
            lines
        }
        crate::EquivalenceResult::False => vec!["False".to_string()],
        crate::EquivalenceResult::Unknown => {
            vec!["Unknown (cannot prove equivalence)".to_string()]
        }
    }
}

/// Format `equiv` command output, including diagnostics when non-equivalence is proven.
pub(crate) fn format_equiv_command_output_lines(
    output: &crate::equiv_command::EquivCommandOutput,
) -> Vec<String> {
    let mut lines = format_equivalence_result_lines(&output.result);
    if let Some(residual) = &output.residual {
        lines.push("Residual:".to_string());
        lines.push(residual.clone());
    }
    lines
}

/// Format parse errors for commands that expect `<expr1>, <expr2>` input.
pub(crate) fn format_expr_pair_parse_error_message(
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

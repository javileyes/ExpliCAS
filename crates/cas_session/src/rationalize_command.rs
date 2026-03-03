//! Session-level helpers for `rationalize` command parsing/formatting.

const RATIONALIZE_USAGE_MESSAGE: &str = "Usage: rationalize <expr>\n\
                 Example: rationalize 1/(1 + sqrt(2) + sqrt(3))";

fn parse_rationalize_input(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

fn format_rationalize_eval_lines(
    context: &cas_ast::Context,
    normalized_expr: cas_ast::ExprId,
    outcome: cas_solver::RationalizeCommandOutcome,
) -> Vec<String> {
    let user_style = cas_formatter::root_style::detect_root_style(context, normalized_expr);
    let parsed = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context,
            id: normalized_expr
        }
    );

    let line = match outcome {
        cas_solver::RationalizeCommandOutcome::Success(simplified_expr) => {
            let style = cas_formatter::root_style::StylePreferences::with_root_style(user_style);
            let rendered = cas_formatter::DisplayExprStyled::new(context, simplified_expr, &style);
            format!("Parsed: {}\nRationalized: {}", parsed, rendered)
        }
        cas_solver::RationalizeCommandOutcome::NotApplicable => format!(
            "Parsed: {}\n\
             Cannot rationalize: denominator is not a sum of surds\n\
             (Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)",
            parsed
        ),
        cas_solver::RationalizeCommandOutcome::BudgetExceeded => format!(
            "Parsed: {}\n\
             Rationalization aborted: expression became too complex",
            parsed
        ),
    };

    vec![line]
}

/// Evaluate `rationalize` command and return final display lines.
pub fn evaluate_rationalize_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
) -> Result<Vec<String>, String> {
    let Some(rest) = parse_rationalize_input(line) else {
        return Err(RATIONALIZE_USAGE_MESSAGE.to_string());
    };

    let output =
        cas_solver::evaluate_rationalize_command_input(simplifier, rest).map_err(|error| {
            match error {
                cas_solver::RationalizeCommandEvalError::Parse(message) => {
                    format!("Parse error: {}", message)
                }
            }
        })?;

    Ok(format_rationalize_eval_lines(
        &simplifier.context,
        output.normalized_expr,
        output.outcome,
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_rationalize_command_lines_empty_input_returns_usage() {
        let mut simplifier = cas_solver::Simplifier::new();
        let err = super::evaluate_rationalize_command_lines(&mut simplifier, "rationalize")
            .expect_err("expected usage");
        assert!(err.contains("Usage: rationalize"));
    }
}

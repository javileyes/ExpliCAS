//! Session-level helpers for `rationalize` command parsing/formatting.

const RATIONALIZE_USAGE_MESSAGE: &str = "Usage: rationalize <expr>\n\
                 Example: rationalize 1/(1 + sqrt(2) + sqrt(3))";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RationalizeCommandOutcome {
    Success(cas_ast::ExprId),
    NotApplicable,
    BudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RationalizeCommandEvalError {
    Parse(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RationalizeCommandEvalOutput {
    normalized_expr: cas_ast::ExprId,
    outcome: RationalizeCommandOutcome,
}

fn parse_rationalize_input(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

fn evaluate_rationalize_command_input(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<RationalizeCommandEvalOutput, RationalizeCommandEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| RationalizeCommandEvalError::Parse(format!("{:?}", e)))?;
    let normalized_expr =
        cas_solver::canonical_forms::normalize_core(&mut simplifier.context, parsed_expr);
    let config = cas_solver::RationalizeConfig::default();
    let rationalized =
        cas_solver::rationalize_denominator(&mut simplifier.context, normalized_expr, &config);
    let outcome = match rationalized {
        cas_solver::RationalizeResult::Success(expr) => {
            RationalizeCommandOutcome::Success(simplifier.simplify(expr).0)
        }
        cas_solver::RationalizeResult::NotApplicable => RationalizeCommandOutcome::NotApplicable,
        cas_solver::RationalizeResult::BudgetExceeded => RationalizeCommandOutcome::BudgetExceeded,
    };
    Ok(RationalizeCommandEvalOutput {
        normalized_expr,
        outcome,
    })
}

fn format_rationalize_eval_lines(
    context: &cas_ast::Context,
    normalized_expr: cas_ast::ExprId,
    outcome: RationalizeCommandOutcome,
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
        RationalizeCommandOutcome::Success(simplified_expr) => {
            let style = cas_formatter::root_style::StylePreferences::with_root_style(user_style);
            let rendered = cas_formatter::DisplayExprStyled::new(context, simplified_expr, &style);
            format!("Parsed: {}\nRationalized: {}", parsed, rendered)
        }
        RationalizeCommandOutcome::NotApplicable => format!(
            "Parsed: {}\n\
             Cannot rationalize: denominator is not a sum of surds\n\
             (Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)",
            parsed
        ),
        RationalizeCommandOutcome::BudgetExceeded => format!(
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
        evaluate_rationalize_command_input(simplifier, rest).map_err(|error| match error {
            RationalizeCommandEvalError::Parse(message) => format!("Parse error: {}", message),
        })?;

    Ok(format_rationalize_eval_lines(
        &simplifier.context,
        output.normalized_expr,
        output.outcome,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_rationalize_command_input, RationalizeCommandEvalError, RationalizeCommandOutcome,
    };

    #[test]
    fn evaluate_rationalize_command_lines_empty_input_returns_usage() {
        let mut simplifier = cas_solver::Simplifier::new();
        let err = super::evaluate_rationalize_command_lines(&mut simplifier, "rationalize")
            .expect_err("expected usage");
        assert!(err.contains("Usage: rationalize"));
    }

    #[test]
    fn evaluate_rationalize_command_input_parse_error_is_typed() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let err =
            evaluate_rationalize_command_input(&mut simplifier, "1/(1+").expect_err("parse error");
        assert!(matches!(err, RationalizeCommandEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_rationalize_command_input_runs() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let out =
            evaluate_rationalize_command_input(&mut simplifier, "1/(1+sqrt(2))").expect("eval");
        match out.outcome {
            RationalizeCommandOutcome::Success(_)
            | RationalizeCommandOutcome::NotApplicable
            | RationalizeCommandOutcome::BudgetExceeded => {}
        }
    }
}

use cas_ast::ExprId;

use crate::Simplifier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RationalizeCommandOutcome {
    Success(ExprId),
    NotApplicable,
    BudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalizeCommandEvalError {
    Parse(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalizeCommandEvalOutput {
    pub normalized_expr: ExprId,
    pub outcome: RationalizeCommandOutcome,
}

pub fn evaluate_rationalize_command_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<RationalizeCommandEvalOutput, RationalizeCommandEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| RationalizeCommandEvalError::Parse(format!("{:?}", e)))?;
    let normalized_expr =
        crate::canonical_forms::normalize_core(&mut simplifier.context, parsed_expr);
    let config = crate::RationalizeConfig::default();
    let rationalized =
        crate::rationalize_denominator(&mut simplifier.context, normalized_expr, &config);
    let outcome = match rationalized {
        crate::RationalizeResult::Success(expr) => {
            RationalizeCommandOutcome::Success(simplifier.simplify(expr).0)
        }
        crate::RationalizeResult::NotApplicable => RationalizeCommandOutcome::NotApplicable,
        crate::RationalizeResult::BudgetExceeded => RationalizeCommandOutcome::BudgetExceeded,
    };
    Ok(RationalizeCommandEvalOutput {
        normalized_expr,
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_rationalize_command_input, RationalizeCommandEvalError, RationalizeCommandOutcome,
    };

    #[test]
    fn evaluate_rationalize_command_input_parse_error_is_typed() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let err =
            evaluate_rationalize_command_input(&mut simplifier, "1/(1+").expect_err("parse error");
        assert!(matches!(err, RationalizeCommandEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_rationalize_command_input_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out =
            evaluate_rationalize_command_input(&mut simplifier, "1/(1+sqrt(2))").expect("eval");
        match out.outcome {
            RationalizeCommandOutcome::Success(_)
            | RationalizeCommandOutcome::NotApplicable
            | RationalizeCommandOutcome::BudgetExceeded => {}
        }
    }
}

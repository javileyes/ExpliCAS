use crate::rationalize_command::{
    RationalizeCommandEvalError, RationalizeCommandEvalOutput, RationalizeCommandOutcome,
};
use cas_math::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};

pub(crate) fn evaluate_rationalize_command_input(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<RationalizeCommandEvalOutput, RationalizeCommandEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| RationalizeCommandEvalError::Parse(format!("{:?}", e)))?;
    let normalized_expr =
        cas_math::canonical_forms::normalize_core(&mut simplifier.context, parsed_expr);
    let config = RationalizeConfig::default();
    let rationalized = rationalize_denominator(&mut simplifier.context, normalized_expr, &config);
    let outcome = match rationalized {
        RationalizeResult::Success(expr) => {
            RationalizeCommandOutcome::Success(simplifier.simplify(expr).0)
        }
        RationalizeResult::NotApplicable => RationalizeCommandOutcome::NotApplicable,
        RationalizeResult::BudgetExceeded => RationalizeCommandOutcome::BudgetExceeded,
    };
    Ok(RationalizeCommandEvalOutput {
        normalized_expr,
        outcome,
    })
}

use crate::rationalize_command_types::{
    RationalizeCommandEvalError, RationalizeCommandEvalOutput, RationalizeCommandOutcome,
};

pub(crate) fn evaluate_rationalize_command_input(
    simplifier: &mut crate::Simplifier,
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

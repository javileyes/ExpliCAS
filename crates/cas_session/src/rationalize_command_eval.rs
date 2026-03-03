use crate::rationalize_command_types::{
    RationalizeCommandEvalError, RationalizeCommandEvalOutput, RationalizeCommandOutcome,
};

pub(crate) fn evaluate_rationalize_command_input(
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

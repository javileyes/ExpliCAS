//! Post-calculus presentation for fractional denominator power integrations.
//!
//! This module keeps the source-side integrand gate close to the result
//! compactors it enables, without changing the integration route itself.

use super::fractional_denominator_power_integrand_preservation::fractional_denominator_power_substitution_integrand_for_calculus_presentation;
use super::integration_conditions::integrate_required_positive_conditions;
use super::negative_half_power_result_presentation::compact_negative_half_power_result_for_integration_presentation;
use super::negative_odd_half_power_result_presentation::compact_negative_three_half_power_result_for_integration_presentation;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn compact_fractional_denominator_power_result_for_integration_presentation(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
) -> Option<ExprId> {
    if fractional_denominator_power_substitution_integrand_for_calculus_presentation(
        ctx,
        call.target,
        &call.var_name,
    ) {
        let allow_conditional_positive_quadratic =
            !integrate_required_positive_conditions(ctx, call.target, &call.var_name).is_empty();
        if let Some(compact) = compact_negative_three_half_power_result_for_integration_presentation(
            ctx,
            result,
            &call.var_name,
            allow_conditional_positive_quadratic,
        ) {
            return Some(compact);
        }
        if let Some(compact) =
            compact_negative_half_power_result_for_integration_presentation(ctx, result)
        {
            return Some(compact);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::symbolic_calculus_call_support::NamedVarCall;
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::compact_fractional_denominator_power_result_for_integration_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn fractional_denominator_power_result_presentation_uses_sqrt_denominator() {
        let mut ctx = Context::new();
        let target = parse("(2*x+1)/(x^2+x+1)^(3/2)", &mut ctx).unwrap();
        let raw_result = parse("-2*(x^2+x+1)^(-1/2)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let compact = compact_fractional_denominator_power_result_for_integration_presentation(
            &mut ctx, &call, raw_result,
        )
        .unwrap();

        assert_eq!(rendered(&ctx, compact), "-2 / sqrt(x^2 + x + 1)");
    }
}

use super::arctan_additive_result_presentation::compact_arctan_additive_terms_for_calculus_presentation;
use super::fractional_denominator_power_integrand_preservation::fractional_denominator_power_substitution_integrand_for_calculus_presentation;
use super::integration::integrate_required_positive_conditions;
use super::negative_half_power_result_presentation::compact_negative_half_power_result_for_integration_presentation;
use super::negative_odd_half_power_result_presentation::compact_negative_three_half_power_result_for_integration_presentation;
use super::positive_half_power_result_presentation::compact_positive_half_power_result_for_integration_presentation;
use super::rationalized_sqrt_result_presentation::compact_acosh_surd_width_arg_for_integration_presentation;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn try_integrate_post_calculus_presentation(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
) -> Option<ExprId> {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
        ctx,
        call.target,
        &call.var_name,
    ) {
        if let Some(compact) =
            compact_arctan_additive_terms_for_calculus_presentation(ctx, result, &call.var_name)
        {
            return Some(compact);
        }
    }
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
    if let Some(compact) =
        compact_positive_half_power_result_for_integration_presentation(ctx, result)
    {
        return Some(compact);
    }
    if let Some(compact) = compact_acosh_surd_width_arg_for_integration_presentation(ctx, result) {
        return Some(compact);
    }
    None
}

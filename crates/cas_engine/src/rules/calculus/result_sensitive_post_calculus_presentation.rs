use cas_ast::{Context, ExprId};

use super::log_root_derivative_presentation::ln_sqrt_negative_polynomial_gap_target;
use super::power_result_presentation::compact_negative_half_power_result_for_integration_presentation;
use super::result_presentation::try_diff_integral_source_post_calculus_presentation;
use super::sqrt_hyperbolic_log_integrand_presentation::{
    compact_direct_sqrt_hyperbolic_log_derivative_integrand, sqrt_cosh_log_derivative_presentation,
};
use super::sqrt_trig_log_antiderivative_presentation::sqrt_trig_log_antiderivative_derivative_presentation;

fn sqrt_log_family_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = sqrt_cosh_log_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if ln_sqrt_negative_polynomial_gap_target(ctx, target, var_name) {
        if let Some(compact) =
            compact_negative_half_power_result_for_integration_presentation(ctx, result)
        {
            return Some(compact);
        }
    }
    if let Some((compact, _)) =
        sqrt_trig_log_antiderivative_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, var_name)
}

pub(super) fn result_sensitive_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        try_diff_integral_source_post_calculus_presentation(ctx, target, result, var_name)
    {
        return Some(compact);
    }

    sqrt_log_family_post_calculus_presentation(ctx, target, result, var_name)
}

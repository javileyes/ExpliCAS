use cas_ast::{Context, ExprId};

use super::ln_sqrt_polynomial_direct_derivative_presentation::ln_sqrt_negative_polynomial_gap_target;
use super::negative_half_power_result_presentation::compact_negative_half_power_result_for_integration_presentation;
use super::sqrt_hyperbolic_log_integrand_presentation::{
    compact_direct_sqrt_hyperbolic_log_derivative_integrand, sqrt_cosh_log_derivative_presentation,
};
use super::sqrt_trig_log_antiderivative_presentation::sqrt_trig_log_antiderivative_derivative_presentation;

pub(super) fn sqrt_log_family_post_calculus_presentation(
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

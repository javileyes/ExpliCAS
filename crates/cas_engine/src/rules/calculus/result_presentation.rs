use super::arctan_additive_result_presentation::compact_arctan_additive_terms_for_calculus_presentation;
use super::half_power_sum_result_presentation::compact_half_power_sum_root_product_for_integration_presentation;
use super::negative_half_power_result_presentation::compact_negative_half_power_product_for_calculus_presentation;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::rationalized_sqrt_result_presentation::compact_acosh_surd_width_arg_for_integration_presentation;
use super::rationalized_sqrt_result_presentation::compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation;
use super::sqrt_hyperbolic_log_integrand_presentation::compact_direct_sqrt_hyperbolic_log_derivative_integrand;
use super::sqrt_hyperbolic_result_presentation::{
    compact_positive_cosh_log_abs_for_integration_presentation, has_compactable_ln_abs_cosh_sqrt,
};
use super::sqrt_trig_log_integrand_presentation::compact_sqrt_trig_log_derivative_integrand;
use super::trig_odd_power_result_presentation::compact_trig_odd_power_reduction_primitive_for_integration_presentation;
use super::trig_result_presentation::compact_trig_square_reduction_primitive_for_integration_presentation;
use cas_ast::{Constant, Context, Expr, ExprId};

pub(crate) fn try_calculus_result_presentation(
    ctx: &mut Context,
    result: ExprId,
) -> Option<ExprId> {
    let result = unwrap_internal_hold_for_calculus(ctx, result);
    if matches!(ctx.get(result), Expr::Constant(Constant::Undefined)) {
        return None;
    }

    compact_sqrt_trig_log_derivative_integrand(ctx, result, "x")
        .or_else(|| compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, "x"))
        .or_else(|| {
            compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation(ctx, result)
        })
        .or_else(|| compact_negative_half_power_product_for_calculus_presentation(ctx, result))
        .or_else(|| {
            has_compactable_ln_abs_cosh_sqrt(ctx, result, "x").then(|| {
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, result, "x")
            })
        })
        .or_else(|| {
            compact_half_power_sum_root_product_for_integration_presentation(ctx, result, "x")
        })
        .or_else(|| {
            compact_trig_square_reduction_primitive_for_integration_presentation(ctx, result, "x")
        })
        .or_else(|| {
            compact_trig_odd_power_reduction_primitive_for_integration_presentation(ctx, result)
        })
        .or_else(|| compact_acosh_surd_width_arg_for_integration_presentation(ctx, result))
        .or_else(|| compact_arctan_additive_terms_for_calculus_presentation(ctx, result, "x"))
}

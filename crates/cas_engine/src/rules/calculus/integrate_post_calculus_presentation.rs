use super::integration_arctan_by_parts_result_presentation::try_compact_polynomial_arctan_by_parts_result_for_post_calculus_presentation;
use super::integration_fractional_power_result_presentation::compact_fractional_denominator_power_result_for_integration_presentation;
use super::positive_half_power_result_presentation::compact_positive_half_power_result_for_integration_presentation;
use super::rationalized_sqrt_result_presentation::compact_acosh_surd_width_arg_for_integration_presentation;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn try_integrate_post_calculus_presentation(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
) -> Option<ExprId> {
    try_compact_polynomial_arctan_by_parts_result_for_post_calculus_presentation(ctx, call, result)
        .or_else(|| {
            compact_fractional_denominator_power_result_for_integration_presentation(
                ctx, call, result,
            )
        })
        .or_else(|| compact_positive_half_power_result_for_integration_presentation(ctx, result))
        .or_else(|| compact_acosh_surd_width_arg_for_integration_presentation(ctx, result))
}

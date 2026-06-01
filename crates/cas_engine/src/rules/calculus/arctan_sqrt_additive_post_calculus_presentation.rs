use cas_ast::{Context, ExprId};

use super::arctan_sqrt_additive_derivative_presentation::{
    arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation,
    arctan_sqrt_additive_tan_polynomial_derivative_presentation,
    arctan_sqrt_additive_trig_polynomial_derivative_presentation,
    arctan_sqrt_small_additive_elementary_derivative_presentation,
};
use super::presentation_utils::unwrap_internal_hold_for_calculus;

pub(super) fn arctan_sqrt_additive_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some((compact, _, _)) =
        arctan_sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }

    if let Some((compact, _, _)) =
        arctan_sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, target, var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }

    if let Some((compact, _, _)) =
        arctan_sqrt_additive_tan_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }

    let (compact, _, _) =
        arctan_sqrt_small_additive_elementary_derivative_presentation(ctx, target, var_name)?;
    Some(unwrap_internal_hold_for_calculus(ctx, compact))
}

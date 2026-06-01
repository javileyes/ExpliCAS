use cas_ast::{Context, ExprId};

use super::acosh_affine_derivative_presentation::constant_scaled_acosh_affine_derivative_presentation;
use super::inverse_reciprocal_trig_affine_abs_derivative_presentation::constant_scaled_inverse_reciprocal_trig_affine_abs_presentation;

pub(super) fn affine_inverse_family_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some((compact, _)) =
        constant_scaled_acosh_affine_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    let (compact, _) =
        constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(ctx, target, var_name)?;
    Some(compact)
}

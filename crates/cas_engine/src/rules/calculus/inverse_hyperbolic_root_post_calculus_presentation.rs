use cas_ast::{Context, ExprId};

use super::acosh_over_sqrt_derivative_presentation::{
    acosh_polynomial_over_sqrt_derivative_presentation,
    constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation,
};
use super::acosh_sqrt_derivative_presentation::acosh_sqrt_family_derivative_presentation;
use super::asinh_sqrt_derivative_presentation::asinh_sqrt_family_derivative_presentation;
use super::atanh_sqrt_derivative_presentation::atanh_sqrt_family_derivative_presentation;

pub(super) fn inverse_hyperbolic_root_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = asinh_sqrt_family_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = atanh_sqrt_family_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = acosh_sqrt_family_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    let (compact, _) =
        constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, var_name)?;
    Some(compact)
}

use cas_ast::{Context, ExprId};

use super::acosh_affine_derivative_presentation::acosh_affine_derivative_presentation;
use super::acosh_strictly_positive_polynomial_derivative_presentation::acosh_strictly_positive_polynomial_derivative_presentation;
use super::asinh_polynomial_derivative_presentation::asinh_polynomial_derivative_presentation;
use super::asinh_sqrt_constant_over_polynomial_presentation::asinh_sqrt_constant_over_polynomial_presentation;
use super::atanh_sqrt_constant_over_polynomial_presentation::atanh_sqrt_constant_over_polynomial_presentation;
use super::inverse_tangent_hyperbolic_rational_affine_derivative_presentation::{
    arctan_rational_affine_derivative_presentation, atanh_rational_affine_derivative_presentation,
};
use super::inverse_trig_derivative_presentation::{
    bounded_inverse_trig_polynomial_derivative_presentation,
    unit_interval_bounded_inverse_trig_derivative_presentation,
};
use super::presentation_utils::unwrap_internal_hold_for_calculus;

pub(super) fn elementary_inverse_function_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        unit_interval_bounded_inverse_trig_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = arctan_rational_affine_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = atanh_rational_affine_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = asinh_polynomial_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) = acosh_affine_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        acosh_strictly_positive_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some((compact, _)) =
        asinh_sqrt_constant_over_polynomial_presentation(ctx, target, var_name)
    {
        return Some(unwrap_internal_hold_for_calculus(ctx, compact));
    }

    let (compact, _) = atanh_sqrt_constant_over_polynomial_presentation(ctx, target, var_name)?;
    Some(unwrap_internal_hold_for_calculus(ctx, compact))
}

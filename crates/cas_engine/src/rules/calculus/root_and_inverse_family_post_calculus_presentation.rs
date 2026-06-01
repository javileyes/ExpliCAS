use cas_ast::{Context, ExprId};

use super::affine_inverse_family_post_calculus_presentation::affine_inverse_family_post_calculus_presentation;
use super::bounded_inverse_root_quotient_post_calculus_presentation::bounded_inverse_root_quotient_post_calculus_presentation;
use super::constant_scaled_inverse_trig_root_post_calculus_presentation::constant_scaled_inverse_trig_root_post_calculus_presentation;
use super::inverse_hyperbolic_root_post_calculus_presentation::inverse_hyperbolic_root_post_calculus_presentation;
use super::inverse_reciprocal_trig_positive_quadratic_derivative_presentation::inverse_reciprocal_trig_post_calculus_presentation;
use super::inverse_surd_quotient_derivative_presentation::inverse_surd_quotient_post_calculus_presentation;
use super::inverse_tangent_scaled_root_quotient_post_calculus_presentation::inverse_tangent_scaled_root_quotient_post_calculus_presentation;
use super::sqrt_derivative_post_calculus_presentation::sqrt_derivative_post_calculus_presentation;

pub(super) fn root_and_inverse_family_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = sqrt_derivative_post_calculus_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_inverse_trig_root_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = affine_inverse_family_post_calculus_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) = inverse_surd_quotient_post_calculus_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_root_quotient_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_tangent_scaled_root_quotient_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = inverse_hyperbolic_root_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    inverse_reciprocal_trig_post_calculus_presentation(ctx, target, var_name)
}

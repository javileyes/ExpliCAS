use cas_ast::{Context, ExprId};

use super::arctan_surd_derivative_presentation::constant_scaled_arctan_surd_quotient_scaled_compact_derivative;
use super::constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation;
use super::inverse_surd_quotient_derivative_presentation::reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative;
use super::inverse_tangent_polynomial_root_derivative_presentation::constant_scaled_arctan_sqrt_polynomial_derivative_presentation;
use super::inverse_tangent_scaled_root_derivative_presentation::{
    constant_scaled_inverse_tangent_scaled_sqrt_polynomial_derivative_presentation,
    inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation,
};

pub(super) fn constant_scaled_inverse_trig_root_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        constant_scaled_arctan_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx, target, var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) =
        inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
            ctx, target, var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) =
        constant_scaled_arctan_surd_quotient_scaled_compact_derivative(ctx, target, var_name)
    {
        return Some(compact);
    }

    let (compact, _) =
        constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
            ctx, target, var_name,
        )?;
    Some(compact)
}

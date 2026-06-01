use cas_ast::{Context, ExprId};
use num_rational::BigRational;
use num_traits::One;

use super::arctan_sqrt_quotient_derivative_presentation::{
    arctan_sqrt_affine_partition_quotient_derivative_presentation,
    arctan_sqrt_polynomial_quotient_derivative_presentation,
};
use super::inverse_tangent_scaled_root_derivative_presentation::inverse_tangent_scaled_sqrt_polynomial_derivative_presentation;

pub(super) fn inverse_tangent_scaled_root_quotient_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        var_name,
        -BigRational::one(),
    ) {
        return Some(compact);
    }
    if let Some(compact) = arctan_sqrt_affine_partition_quotient_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    ) {
        return Some(compact);
    }

    arctan_sqrt_polynomial_quotient_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    )
}

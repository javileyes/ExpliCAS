//! Source-side gates for inverse square-root product integration families.
//!
//! Keep these predicates separate: the arcsin inverse-product and affine
//! sqrt-product routes share compact source presentation, but they carry
//! different primitive and domain policies.

use cas_ast::{Context, ExprId};

pub(super) fn arcsin_inverse_sqrt_product_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
        ctx, target, var_name,
    )
}

pub(super) fn affine_sqrt_product_derivative_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_affine_sqrt_product_derivative_target(
        ctx, target, var_name,
    )
}

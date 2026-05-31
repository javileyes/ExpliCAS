//! Source-side hyperbolic power integrand detection.

use cas_ast::{Context, ExprId};

pub(super) fn hyperbolic_power_integrand_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let preserve_compact_affine_hyperbolic_square =
        cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_square_target(
            ctx, expr, var_name,
        );
    let preserve_compact_affine_hyperbolic_cubic =
        cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_cubic_target(
            ctx, expr, var_name,
        );
    let preserve_compact_affine_hyperbolic_fifth =
        cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_fifth_target(
            ctx, expr, var_name,
        );
    let preserve_compact_affine_hyperbolic_seventh =
        cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_seventh_target(
            ctx, expr, var_name,
        );
    let preserve_compact_hyperbolic_square_product =
        cas_math::symbolic_integration_support::integrate_symbolic_is_hyperbolic_square_product_target(
            ctx, expr, var_name,
        );

    preserve_compact_affine_hyperbolic_square
        || preserve_compact_affine_hyperbolic_cubic
        || preserve_compact_affine_hyperbolic_fifth
        || preserve_compact_affine_hyperbolic_seventh
        || preserve_compact_hyperbolic_square_product
}

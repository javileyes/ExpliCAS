//! Source-side affine inverse-hyperbolic integrand preservation gates.

use cas_ast::{Context, ExprId};

pub(super) fn inverse_hyperbolic_affine_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let preserve_compact_asinh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_affine_variable_target(
            ctx, target, var_name,
        );
    let preserve_compact_atanh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_atanh_affine_variable_target(
            ctx, target, var_name,
        );
    let preserve_compact_acosh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_affine_variable_target(
            ctx, target, var_name,
        );

    preserve_compact_asinh_affine || preserve_compact_atanh_affine || preserve_compact_acosh_affine
}

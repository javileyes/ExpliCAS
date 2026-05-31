//! Source-side arctan integrand preservation gates.

use cas_ast::{Context, ExprId};

pub(super) fn arctan_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let preserve_compact_arctan_reciprocal_affine = cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_reciprocal_affine_variable_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_arctan_sqrt_reciprocal = cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_arctan_sqrt_unit_shift_square = cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_unit_shift_square_target(
        ctx,
        target,
        var_name,
    );
    let preserve_compact_arctan_scaled_variable =
        cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_scaled_variable_target(
            ctx, target, var_name,
        );

    preserve_compact_arctan_reciprocal_affine
        || preserve_compact_arctan_sqrt_reciprocal
        || preserve_compact_arctan_sqrt_unit_shift_square
        || preserve_compact_arctan_scaled_variable
}

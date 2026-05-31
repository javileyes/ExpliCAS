//! Source-side logarithmic by-parts integrand detection.

use cas_ast::{Context, Expr, ExprId};

pub(super) fn log_by_parts_integrand_for_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    if !crate::rule::steps_enabled() {
        return false;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_monomial_times_ln_var_by_parts_target(
        ctx, target, var_name,
    ) {
        return true;
    }

    if matches!(ctx.get(target), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_affine_ln_by_parts_target(
        ctx, target, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
        ctx, target, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_positive_quadratic_ln_by_parts_target(
        ctx, target, var_name,
    )
}

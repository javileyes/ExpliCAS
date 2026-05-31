//! Source-side exponential by-parts integrand detection.

use cas_ast::{Context, ExprId};

pub(super) fn linear_exp_by_parts_integrand_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    crate::rule::steps_enabled()
        && cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_exp_linear_target(
            ctx, expr, var_name,
        )
}

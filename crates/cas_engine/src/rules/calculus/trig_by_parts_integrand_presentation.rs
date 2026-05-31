//! Source-side trigonometric by-parts integrand detection.

use cas_ast::{Context, Expr, ExprId};

pub(super) fn linear_trig_by_parts_integrand_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let allow_presentation =
        crate::rule::steps_enabled() || target_has_top_level_negative_orientation(ctx, expr);
    allow_presentation
        && cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_trig_linear_target(
            ctx, expr, var_name,
        )
}

pub(super) fn repeated_trig_by_parts_integrand_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_trig_linear_target(
        ctx, expr, var_name,
    )
}

fn target_has_top_level_negative_orientation(ctx: &Context, target: ExprId) -> bool {
    match ctx.get(target) {
        Expr::Neg(_) => true,
        Expr::Mul(left, right) => {
            matches!(ctx.get(*left), Expr::Neg(_)) || matches!(ctx.get(*right), Expr::Neg(_))
        }
        _ => false,
    }
}

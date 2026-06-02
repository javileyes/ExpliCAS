use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_utils::{calculus_sqrt_like_radicand, sqrt_raw_for_calculus_presentation};
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;

pub(super) fn compact_shifted_sqrt_argument_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(radicand) = calculus_sqrt_like_radicand(ctx, expr) {
        if !contains_named_var(ctx, radicand, var_name) {
            return None;
        }
        polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
        return Some(sqrt_raw_for_calculus_presentation(ctx, radicand));
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => compact_sqrt_shift_with_constant_side(ctx, left, right, var_name)
            .map(|(left, right)| ctx.add_raw(Expr::Add(left, right))),
        Expr::Sub(left, right) => compact_sqrt_shift_with_constant_side(ctx, left, right, var_name)
            .map(|(left, right)| ctx.add_raw(Expr::Sub(left, right))),
        _ => None,
    }
}

fn compact_sqrt_shift_with_constant_side(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    if !contains_named_var(ctx, right, var_name) {
        if let Some(compact_left) =
            compact_shifted_sqrt_argument_for_integration_presentation(ctx, left, var_name)
        {
            return Some((compact_left, right));
        }
    }

    if !contains_named_var(ctx, left, var_name) {
        if let Some(compact_right) =
            compact_shifted_sqrt_argument_for_integration_presentation(ctx, right, var_name)
        {
            return Some((left, compact_right));
        }
    }

    None
}

use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_utils::calculus_sqrt_like_radicand;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;

pub(super) fn compact_sqrt_trig_log_abs_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(compact) = compact_ln_abs_trig_sqrt(ctx, expr, var_name) {
        return compact;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, num, var_name);
            let den = compact_sqrt_trig_log_abs_for_integration_presentation(ctx, den, var_name);
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner =
                compact_sqrt_trig_log_abs_for_integration_presentation(ctx, inner, var_name);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn compact_ln_abs_trig_sqrt(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let Expr::Function(abs_fn, abs_args) = ctx.get(args[0]).clone() else {
        return None;
    };
    if ctx.builtin_of(abs_fn) != Some(BuiltinFn::Abs) || abs_args.len() != 1 {
        return None;
    }

    let Expr::Function(trig_fn, trig_args) = ctx.get(abs_args[0]).clone() else {
        return None;
    };
    let trig_builtin = ctx.builtin_of(trig_fn)?;
    if !matches!(trig_builtin, BuiltinFn::Sin | BuiltinFn::Cos) || trig_args.len() != 1 {
        return None;
    }

    let radicand = calculus_sqrt_like_radicand(ctx, trig_args[0])?;
    if !contains_named_var(ctx, radicand, var_name) {
        return None;
    }
    polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig_expr = ctx.call_builtin(trig_builtin, vec![sqrt_radicand]);
    let abs_expr = ctx.call_builtin(BuiltinFn::Abs, vec![trig_expr]);
    Some(ctx.call_builtin(BuiltinFn::Ln, vec![abs_expr]))
}

pub(super) fn has_compactable_ln_abs_trig_sqrt(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_ln_abs_trig_sqrt(ctx, expr, var_name).is_some() {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_ln_abs_trig_sqrt(ctx, left, var_name)
                || has_compactable_ln_abs_trig_sqrt(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_ln_abs_trig_sqrt(ctx, num, var_name)
                || has_compactable_ln_abs_trig_sqrt(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_ln_abs_trig_sqrt(ctx, inner, var_name),
        _ => false,
    }
}

//! Shared unary-function shape helpers for calculus presentation routes.

use super::presentation_utils::same_sqrt_like_argument;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

pub(super) fn unary_builtin_arg_for_calculus(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin)).then_some(args[0])
}

pub(super) fn signed_unary_builtin_arg_for_calculus(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, BigRational)> {
    if let Some(arg) = unary_builtin_arg_for_calculus(ctx, expr, builtin) {
        return Some((arg, BigRational::one()));
    }

    if let Expr::Neg(inner) = ctx.get(expr) {
        return unary_builtin_arg_for_calculus(ctx, *inner, builtin)
            .map(|arg| (arg, -BigRational::one()));
    }

    if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
        return None;
    }

    let mut scale = BigRational::one();
    let mut arg = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        let factor_arg = unary_builtin_arg_for_calculus(ctx, factor, builtin)?;
        if arg.replace(factor_arg).is_some() {
            return None;
        }
    }

    (scale == -BigRational::one()).then_some((arg?, -BigRational::one()))
}

pub(super) fn unordered_same_sqrt_like_unary_pair_for_calculus(
    ctx: &mut Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    same_sqrt_like_unary_pair_for_calculus(ctx, left, left_builtin, right, right_builtin).or_else(
        || same_sqrt_like_unary_pair_for_calculus(ctx, right, left_builtin, left, right_builtin),
    )
}

pub(super) fn same_sqrt_like_unary_pair_for_calculus(
    ctx: &mut Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    let left_arg = unary_builtin_arg_for_calculus(ctx, left, left_builtin)?;
    let right_arg = unary_builtin_arg_for_calculus(ctx, right, right_builtin)?;
    same_sqrt_like_argument(ctx, left_arg, right_arg).then_some(left_arg)
}

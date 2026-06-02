use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};

use super::presentation_utils::{calculus_sqrt_like_radicand, is_half_power_exponent};

pub(super) fn positive_constant_over_expr_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            let value = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            value.is_positive().then_some((value, *den))
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            Some((BigRational::one(), *base))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut denominator = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 1.into()))
                    || denominator.replace(*base).is_some()
                {
                    return None;
                }
            }
            scale.is_positive().then_some((scale, denominator?))
        }
        _ => None,
    }
}

pub(super) fn positive_constant_over_inverse_sqrt_arg_for_calculus_presentation(
    ctx: &Context,
    arg: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(arg) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            positive_constant_over_expr_for_calculus_presentation(ctx, sqrt_args[0])
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => {
            positive_constant_over_expr_for_calculus_presentation(ctx, *base)
        }
        Expr::Function(abs_fn, abs_args)
            if abs_args.len() == 1 && ctx.is_builtin(*abs_fn, BuiltinFn::Abs) =>
        {
            positive_constant_over_reciprocal_sqrt_expr_for_calculus_presentation(ctx, abs_args[0])
        }
        _ => None,
    }
}

fn positive_constant_over_reciprocal_sqrt_expr_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            Some((BigRational::one(), *base))
        }
        Expr::Div(num, den) => {
            let numerator = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            if !numerator.is_positive() {
                return None;
            }
            let denominator = calculus_sqrt_like_radicand(ctx, *den)?;
            Some((numerator, denominator))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut denominator = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 2.into()))
                    || denominator.replace(*base).is_some()
                {
                    return None;
                }
            }
            if !scale.is_positive() {
                return None;
            }
            let scale_squared = &scale * &scale;
            Some((scale_squared, denominator?))
        }
        _ => None,
    }
}

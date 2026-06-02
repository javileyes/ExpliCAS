use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::presentation_utils::{is_half_power_exponent, rational_const_for_hold};
use super::scalar_presentation::negate_calculus_presentation;
use super::scalar_rational_sqrt_presentation::{
    exact_positive_rational_sqrt_for_calculus_presentation,
    sqrt_positive_rational_expr_for_calculus_presentation,
};
use super::scalar_scale_presentation::signed_numerator_for_calculus_presentation;

pub(super) fn fold_numeric_mul_constants_for_hold(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(_, _) => {
            let mut factors = cas_math::expr_nary::mul_leaves(ctx, expr);
            let mut scale = BigRational::one();
            let mut non_numeric = Vec::new();

            while let Some(factor) = factors.pop() {
                let folded = fold_numeric_mul_constants_for_hold(ctx, factor);
                if matches!(ctx.get(folded), Expr::Mul(_, _)) {
                    factors.extend(cas_math::expr_nary::mul_leaves(ctx, folded));
                    continue;
                }
                if let Some(value) = rational_const_for_hold(ctx, folded) {
                    scale *= value;
                } else {
                    non_numeric.push(folded);
                }
            }

            if scale.is_zero() {
                return ctx.num(0);
            }

            if !scale.is_one() && non_numeric.len() == 1 {
                if let Some(radicand) = sqrt_positive_rational_for_hold(ctx, non_numeric[0]) {
                    let sign = if scale.is_negative() {
                        -BigRational::one()
                    } else {
                        BigRational::one()
                    };
                    let scaled_radicand = &scale * &scale * radicand;
                    let folded = if let Some(root) =
                        exact_positive_rational_sqrt_for_calculus_presentation(&scaled_radicand)
                    {
                        ctx.add(Expr::Number(sign * root))
                    } else {
                        let sqrt = sqrt_positive_rational_expr_for_calculus_presentation(
                            ctx,
                            scaled_radicand,
                        );
                        if sign.is_negative() {
                            negate_calculus_presentation(ctx, sqrt)
                        } else {
                            sqrt
                        }
                    };
                    return folded;
                }
                if let Expr::Div(num, den) = ctx.get(non_numeric[0]).clone() {
                    let scale_expr = ctx.add(Expr::Number(scale));
                    let scaled_num = ctx.add(Expr::Mul(scale_expr, num));
                    let folded_num = fold_numeric_mul_constants_for_hold(ctx, scaled_num);
                    return ctx.add(Expr::Div(folded_num, den));
                }
                if scale == -BigRational::one() {
                    if let Expr::Neg(inner) = ctx.get(non_numeric[0]).clone() {
                        return inner;
                    }
                }
            }

            if !scale.is_one() || non_numeric.is_empty() {
                non_numeric.insert(0, ctx.add(Expr::Number(scale)));
            }

            if non_numeric.len() == 1 {
                non_numeric[0]
            } else {
                cas_math::expr_nary::build_balanced_mul(ctx, &non_numeric)
            }
        }
        Expr::Div(num, den) => {
            let num = fold_numeric_mul_constants_for_hold(ctx, num);
            let den = fold_numeric_mul_constants_for_hold(ctx, den);
            if let Some(den_value) = rational_const_for_hold(ctx, den) {
                if den_value.is_zero() {
                    return ctx.add(Expr::Div(num, den));
                }
                if let Some(num_value) = rational_const_for_hold(ctx, num) {
                    return ctx.add(Expr::Number(num_value / den_value));
                }
                let reciprocal = ctx.add(Expr::Number(BigRational::one() / den_value));
                let scaled = ctx.add(Expr::Mul(reciprocal, num));
                return fold_numeric_mul_constants_for_hold(ctx, scaled);
            }
            if let Some(num_value) = rational_const_for_hold(ctx, num)
                .filter(|_| matches!(ctx.get(den), Expr::Mul(_, _)))
            {
                let mut denominator_scale = BigRational::one();
                let mut denominator_factors = Vec::new();
                for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        denominator_scale *= value;
                    } else {
                        denominator_factors.push(factor);
                    }
                }
                if !denominator_scale.is_zero()
                    && !denominator_scale.is_one()
                    && !denominator_factors.is_empty()
                {
                    let scaled_num_value = num_value / denominator_scale;
                    if !scaled_num_value.is_integer() {
                        return ctx.add(Expr::Div(num, den));
                    }
                    let num = ctx.add(Expr::Number(scaled_num_value));
                    let den = if denominator_factors.len() == 1 {
                        denominator_factors[0]
                    } else {
                        cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors)
                    };
                    return ctx.add(Expr::Div(num, den));
                }
            }
            if matches!(ctx.get(den), Expr::Mul(_, _)) {
                let mut numerator_scale = BigRational::one();
                let mut numerator_factors = Vec::new();
                let mut raw_numerator_factors = Vec::new();
                mul_leaves_preserve_order_for_calculus_presentation(
                    ctx,
                    num,
                    &mut raw_numerator_factors,
                );
                for factor in raw_numerator_factors {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        numerator_scale *= value;
                    } else {
                        numerator_factors.push(factor);
                    }
                }

                let mut denominator_scale = BigRational::one();
                let mut denominator_factors = Vec::new();
                let mut raw_denominator_factors = Vec::new();
                mul_leaves_preserve_order_for_calculus_presentation(
                    ctx,
                    den,
                    &mut raw_denominator_factors,
                );
                for factor in raw_denominator_factors {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        denominator_scale *= value;
                    } else {
                        denominator_factors.push(factor);
                    }
                }

                if !denominator_scale.is_zero()
                    && !denominator_scale.is_one()
                    && !numerator_factors.is_empty()
                    && !denominator_factors.is_empty()
                {
                    let scaled_numerator = numerator_scale / denominator_scale;
                    if scaled_numerator.is_integer() {
                        let numerator_core = if numerator_factors.len() == 1 {
                            numerator_factors[0]
                        } else {
                            cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors)
                        };
                        let numerator = signed_numerator_for_calculus_presentation(
                            ctx,
                            scaled_numerator,
                            numerator_core,
                        );
                        let denominator = if denominator_factors.len() == 1 {
                            denominator_factors[0]
                        } else {
                            cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors)
                        };
                        return ctx.add(Expr::Div(numerator, denominator));
                    }
                }
            }
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner = fold_numeric_mul_constants_for_hold(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn mul_leaves_preserve_order_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    out: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            mul_leaves_preserve_order_for_calculus_presentation(ctx, left, out);
            mul_leaves_preserve_order_for_calculus_presentation(ctx, right, out);
        }
        _ => out.push(expr),
    }
}

pub(super) fn fold_numeric_mul_constants_for_hold_additive_terms(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = fold_numeric_mul_constants_for_hold_additive_terms(ctx, left);
            let right = fold_numeric_mul_constants_for_hold_additive_terms(ctx, right);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = fold_numeric_mul_constants_for_hold_additive_terms(ctx, left);
            let right = fold_numeric_mul_constants_for_hold_additive_terms(ctx, right);
            ctx.add(Expr::Sub(left, right))
        }
        _ => fold_numeric_mul_constants_for_hold(ctx, expr),
    }
}

fn sqrt_positive_rational_for_hold(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let value = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            cas_ast::views::as_rational_const(ctx, args[0], 8)?
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => {
            cas_ast::views::as_rational_const(ctx, *base, 8)?
        }
        _ => return None,
    };
    value.is_positive().then_some(value)
}

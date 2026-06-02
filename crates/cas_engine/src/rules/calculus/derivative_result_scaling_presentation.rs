//! Shared post-calculus derivative result scaling helpers.
//!
//! These helpers finish already-matched compact derivative routes. They do not
//! own family detection or domain policy.

use super::polynomial_support::split_polynomial_content_for_calculus_presentation;
use super::presentation_utils::{same_sqrt_like_argument, unwrap_internal_hold_for_calculus};
use super::scalar_presentation::{
    fold_numeric_mul_constants_for_hold, negate_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    scale_fraction_for_calculus_presentation, signed_rational_const_for_calculus_presentation,
};
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn compact_division_by_positive_denominator_content_for_calculus_presentation(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<ExprId> {
    let numerator_value = signed_rational_const_for_calculus_presentation(ctx, num)?;
    let (den_core, den_content) = split_polynomial_content_for_calculus_presentation(ctx, den);
    if !den_content.is_positive() || den_content.is_one() || den_core == den {
        return None;
    }

    let scaled_numerator = numerator_value / den_content;
    let numerator = rational_const_for_calculus_presentation(
        ctx,
        BigRational::from_integer(scaled_numerator.numer().clone()),
    );
    if scaled_numerator.denom().is_one() {
        return Some(ctx.add(Expr::Div(numerator, den_core)));
    }

    let denominator_scale = BigRational::from_integer(scaled_numerator.denom().clone());
    let denominator = scale_expr_for_calculus_presentation(ctx, denominator_scale, den_core);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn cancel_denominator_content_with_numerator_for_calculus_presentation(
    ctx: &mut Context,
    numerator_coeff: BigRational,
    denominator_coeff: BigRational,
    denominator_factor: ExprId,
) -> (BigRational, BigRational, ExprId) {
    let (primitive_factor, factor_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator_factor);
    if factor_content.is_zero() || factor_content.is_one() {
        return (numerator_coeff, denominator_coeff, denominator_factor);
    }

    let adjusted_numerator = numerator_coeff.clone() / factor_content.clone();
    if !adjusted_numerator.is_integer() {
        let adjusted_denominator = denominator_coeff * factor_content;
        let (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / adjusted_denominator))
                .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
        return (numerator_coeff, denominator_coeff, primitive_factor);
    }

    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(adjusted_numerator / denominator_coeff))
            .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    (numerator_coeff, denominator_coeff, primitive_factor)
}

pub(super) fn cancel_positive_denominator_content_with_numerator_for_calculus_presentation(
    ctx: &mut Context,
    numerator_coeff: BigRational,
    denominator_coeff: BigRational,
    denominator_factor: ExprId,
) -> (BigRational, BigRational, ExprId) {
    let (primitive_factor, factor_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator_factor);
    if factor_content.is_zero() || !factor_content.is_positive() || factor_content.is_one() {
        return (numerator_coeff, denominator_coeff, denominator_factor);
    }

    let adjusted_numerator = numerator_coeff.clone() / factor_content.clone();
    if !adjusted_numerator.is_integer() {
        let adjusted_denominator = denominator_coeff * factor_content;
        let (numerator_coeff, denominator_coeff) =
            nonzero_rational_parts(&(numerator_coeff / adjusted_denominator))
                .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
        return (numerator_coeff, denominator_coeff, primitive_factor);
    }

    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(adjusted_numerator / denominator_coeff))
            .unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    (numerator_coeff, denominator_coeff, primitive_factor)
}

pub(super) fn remove_unit_mul_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return expr;
    };

    let mut non_unit_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if cas_ast::views::as_rational_const(ctx, factor, 8) == Some(BigRational::one()) {
            continue;
        }
        non_unit_factors.push(factor);
    }

    match non_unit_factors.as_slice() {
        [single] => *single,
        _ => expr,
    }
}

pub(super) fn scale_compact_derivative_by_rational(
    ctx: &mut Context,
    derivative: ExprId,
    scale: BigRational,
) -> ExprId {
    if scale.is_one() {
        return derivative;
    }

    let derivative = unwrap_internal_hold_for_calculus(ctx, derivative);
    let scaled = if let Expr::Div(num, den) = ctx.get(derivative).clone() {
        let (num, den) = scale_fraction_for_calculus_presentation(ctx, num, den, scale);
        if let Some(compact) =
            compact_division_by_positive_denominator_content_for_calculus_presentation(
                ctx, num, den,
            )
        {
            return compact;
        }
        ctx.add(Expr::Div(num, den))
    } else {
        scale_expr_for_calculus_presentation(ctx, scale, derivative)
    };

    fold_numeric_mul_constants_for_hold(ctx, scaled)
}

pub(super) fn reciprocal_constant_denominator_for_calculus_presentation(
    ctx: &mut Context,
    factor: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if contains_named_var(ctx, factor, var_name) {
        return None;
    }

    match ctx.get(factor).clone() {
        Expr::Number(value) if value.numer() == &BigInt::from(1) && !value.is_zero() => {
            Some(ctx.add(Expr::Number(BigRational::from_integer(
                value.denom().clone(),
            ))))
        }
        Expr::Div(numerator, denominator) => {
            let numerator_value = cas_ast::views::as_rational_const(ctx, numerator, 8)?;
            if numerator_value == BigRational::one() {
                Some(denominator)
            } else {
                None
            }
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            Some(base)
        }
        _ => None,
    }
}

pub(super) fn divide_compact_derivative_by_constant_factor(
    ctx: &mut Context,
    derivative: ExprId,
    outer_den: ExprId,
) -> ExprId {
    let derivative = unwrap_internal_hold_for_calculus(ctx, derivative);
    if let Expr::Div(num, den) = ctx.get(derivative).clone() {
        if let Some(cancelled_num) = remove_matching_sqrt_like_product_factor(ctx, num, outer_den) {
            return ctx.add(Expr::Div(cancelled_num, den));
        }

        let combined_den = cas_math::expr_nary::build_balanced_mul(ctx, &[outer_den, den]);
        return ctx.add(Expr::Div(num, combined_den));
    }

    ctx.add(Expr::Div(derivative, outer_den))
}

fn remove_matching_sqrt_like_product_factor(
    ctx: &mut Context,
    expr: ExprId,
    factor: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let cancelled = remove_matching_sqrt_like_product_factor(ctx, inner, factor)?;
        return Some(negate_calculus_presentation(ctx, cancelled));
    }

    if same_sqrt_like_argument(ctx, expr, factor) {
        return Some(ctx.num(1));
    }

    let mut factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    for idx in 0..factors.len() {
        if same_sqrt_like_argument(ctx, factors[idx], factor) {
            factors.remove(idx);
            return Some(match factors.as_slice() {
                [] => ctx.num(1),
                [single] => *single,
                _ => cas_math::expr_nary::build_balanced_mul(ctx, &factors),
            });
        }
    }

    None
}

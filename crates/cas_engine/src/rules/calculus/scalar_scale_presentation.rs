use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};

pub(super) fn nonzero_rational_parts(value: &BigRational) -> Option<(BigRational, BigRational)> {
    if value.is_zero() {
        return None;
    }

    let numerator = BigRational::from_integer(value.numer().clone());
    let denominator = BigRational::from_integer(value.denom().clone());
    Some((numerator, denominator))
}

pub(super) fn rational_scaled_single_factor_allow_unit(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (inner_scale, inner_factor) = rational_scaled_single_factor_allow_unit(ctx, inner)
            .unwrap_or_else(|| (BigRational::one(), inner));
        return Some((-inner_scale, inner_factor));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut non_numeric = Vec::new();

    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else {
            non_numeric.push(factor);
        }
    }

    let [inner] = non_numeric.as_slice() else {
        return None;
    };

    Some((scale, *inner))
}

pub(super) fn rational_scaled_single_factor(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (inner_scale, inner_factor) = rational_scaled_single_factor_allow_unit(ctx, inner)
            .unwrap_or_else(|| (BigRational::one(), inner));
        let scale = -inner_scale;
        if scale.is_one() {
            return None;
        }
        return Some((scale, inner_factor));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut non_numeric = Vec::new();

    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else {
            non_numeric.push(factor);
        }
    }

    if scale.is_one() {
        return None;
    }

    let [inner] = non_numeric.as_slice() else {
        return None;
    };

    Some((scale, *inner))
}

pub(super) fn split_numeric_scale_single_core(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if den_scale.is_zero() {
            return None;
        }
        let (num_scale, core) = split_numeric_scale_single_core(ctx, num)?;
        return Some((num_scale / den_scale, core));
    }
    let mut scale = BigRational::one();
    let mut core = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if core.replace(factor).is_some() {
            return None;
        }
    }
    Some((scale, core.unwrap_or(expr)))
}

pub(super) fn split_outer_numeric_mul_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let mut scale = BigRational::one();
    let mut core = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if core.replace(factor).is_some() {
            return None;
        }
    }
    Some((scale, core.unwrap_or(expr)))
}

pub(super) fn scale_compact_fraction_numerator_by_rational_for_calculus_presentation(
    ctx: &mut Context,
    derivative: ExprId,
    scale: BigRational,
) -> ExprId {
    if scale.is_one() {
        return unwrap_internal_hold_for_calculus(ctx, derivative);
    }
    if scale.is_zero() {
        return ctx.num(0);
    }

    let derivative = unwrap_internal_hold_for_calculus(ctx, derivative);
    let Expr::Div(num, den) = ctx.get(derivative).clone() else {
        return scale_expr_for_calculus_presentation(ctx, scale, derivative);
    };

    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&scale).unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, num);
    let denominator = if denominator_coeff == BigRational::one() {
        den
    } else {
        scale_expr_for_calculus_presentation(ctx, denominator_coeff, den)
    };

    ctx.add(Expr::Div(numerator, denominator))
}

pub(super) fn scale_fraction_for_calculus_presentation(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    scale: BigRational,
) -> (ExprId, ExprId) {
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&scale).unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator);
    let denominator = if denominator_coeff.is_one() {
        denominator
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, denominator])
    };

    (numerator, denominator)
}

pub(super) fn signed_numerator_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    expr: ExprId,
) -> ExprId {
    if coeff == -BigRational::one()
        && !cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| value.is_one())
    {
        return ctx.add(Expr::Neg(expr));
    }
    scale_expr_for_calculus_presentation(ctx, coeff, expr)
}

pub(super) fn signed_rational_const_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        return Some(value);
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        return cas_ast::views::as_rational_const(ctx, *inner, 8).map(|value| -value);
    }
    None
}

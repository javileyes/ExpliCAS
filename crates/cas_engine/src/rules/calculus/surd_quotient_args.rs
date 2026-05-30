use cas_ast::{Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed};

use super::gap_presentation::reciprocal_positive_rational;
use super::scalar_presentation::rational_const_for_calculus_presentation;

pub(super) fn atanh_arg_over_sqrt_parts(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(arg) {
        Expr::Div(num, den) => {
            let radicand = extract_square_root_base(ctx, *den)?;
            Some((*num, radicand))
        }
        Expr::Mul(_, _) => rationalized_arg_over_sqrt_parts(ctx, arg),
        _ => None,
    }
}

pub(super) fn sqrt_scaled_arg_over_sqrt_parts_for_calculus_presentation(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    if factors.len() < 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().enumerate() {
        let Some(scale_radicand) = extract_square_root_base(ctx, *sqrt_factor) else {
            continue;
        };
        let scale_radicand_value = cas_ast::views::as_rational_const(ctx, scale_radicand, 8)?;
        if !scale_radicand_value.is_positive() {
            return None;
        }

        let mut rational_scale = BigRational::one();
        let mut numerator_factors = Vec::new();
        for (factor_index, factor) in factors.iter().enumerate() {
            if factor_index == sqrt_index {
                continue;
            }

            if let Some(value) = cas_ast::views::as_rational_const(ctx, *factor, 8) {
                rational_scale *= value;
            } else {
                numerator_factors.push(*factor);
            }
        }

        if !rational_scale.is_positive() || numerator_factors.is_empty() {
            continue;
        }

        let scale_square = &rational_scale * &rational_scale;
        let equivalent_denominator_radicand =
            reciprocal_positive_rational(&(scale_square * scale_radicand_value));
        let numerator = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
        let radicand =
            rational_const_for_calculus_presentation(ctx, equivalent_denominator_radicand);
        return Some((numerator, radicand));
    }

    None
}

fn arg_over_scaled_sqrt_parts(ctx: &mut Context, arg: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Div(num, den) = ctx.get(arg).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, den);
    if factors.len() < 2 {
        return None;
    }

    let mut rational_scale = BigRational::one();
    let mut radicand = None;
    for factor in factors {
        if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
            continue;
        }
        let value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        rational_scale *= value;
    }

    if !rational_scale.is_positive() {
        return None;
    }
    let scale_square = &rational_scale * &rational_scale;
    let scale_square_expr = rational_const_for_calculus_presentation(ctx, scale_square);
    let radicand = ctx.add(Expr::Mul(scale_square_expr, radicand?));
    Some((num, radicand))
}

fn rationalized_arg_over_sqrt_parts(ctx: &mut Context, arg: ExprId) -> Option<(ExprId, ExprId)> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    if factors.len() < 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().enumerate() {
        let Some(radicand) = extract_square_root_base(ctx, *sqrt_factor) else {
            continue;
        };
        let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
        if !radicand_value.is_positive() {
            return None;
        }

        let mut rational_scale = BigRational::one();
        let mut numerator_factors = Vec::new();
        for (factor_index, factor) in factors.iter().enumerate() {
            if factor_index == sqrt_index {
                continue;
            }

            if let Some(value) = cas_ast::views::as_rational_const(ctx, *factor, 8) {
                rational_scale *= value;
            } else {
                numerator_factors.push(*factor);
            }
        }

        if numerator_factors.is_empty() {
            continue;
        }

        if rational_scale * &radicand_value == BigRational::one() {
            let numerator = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
            return Some((numerator, radicand));
        }
    }

    None
}

pub(super) fn arctan_self_normalized_surd_quotient_parts(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let Some(parts) = atanh_arg_over_sqrt_parts(ctx, arg) {
        return Some(parts);
    }
    if let Some(parts) = arg_over_scaled_sqrt_parts(ctx, arg) {
        return Some(parts);
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    if factors.len() < 2 {
        return None;
    }

    let neg_half = BigRational::new((-1).into(), 2.into());
    let mut radicand = None;
    let mut numerator_factors = Vec::new();
    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(base, exp)
                if cas_ast::views::as_rational_const(ctx, *exp, 8) == Some(neg_half.clone()) =>
            {
                if radicand.replace(*base).is_some() {
                    return None;
                }
            }
            _ => numerator_factors.push(factor),
        }
    }

    let radicand = radicand?;
    if numerator_factors.is_empty() {
        return None;
    }
    let numerator = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
    Some((numerator, radicand))
}

pub(super) fn arctan_self_normalized_surd_reciprocal_parts(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(arg).clone() {
        Expr::Div(numerator, denominator) => {
            let radicand = extract_square_root_base(ctx, numerator)?;
            Some((denominator, radicand))
        }
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
            let mut radicand = None;
            let mut denominator = None;
            let neg_one = BigRational::new((-1).into(), 1.into());

            for factor in factors {
                if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
                    if radicand.replace(factor_radicand).is_some() {
                        return None;
                    }
                    continue;
                }
                match ctx.get(factor) {
                    Expr::Pow(base, exp)
                        if cas_ast::views::as_rational_const(ctx, *exp, 8)
                            == Some(neg_one.clone()) =>
                    {
                        if denominator.replace(*base).is_some() {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }

            Some((denominator?, radicand?))
        }
        _ => None,
    }
}

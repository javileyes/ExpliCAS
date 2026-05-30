use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::polynomial_support::{
    polynomial_positive_content_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
};
use super::presentation_utils::{
    calculus_sqrt_like_radicand, scaled_sqrt_argument_for_calculus_presentation,
};
use super::scalar_presentation::{
    exact_positive_rational_sqrt_for_calculus_presentation,
    rational_scaled_single_factor_allow_unit,
};

pub(super) fn scaled_sqrt_polynomial_arg_for_calculus_presentation(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial, BigRational)> {
    let (radicand, mut sqrt_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, arg)?;
    let mut radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;

    if let Some(content) = polynomial_positive_content_for_calculus_presentation(&radicand_poly)
        .filter(|content| !content.is_one())
    {
        let sqrt_content = exact_positive_rational_sqrt_for_calculus_presentation(&content)?;
        let primitive_coeffs = radicand_poly
            .coeffs
            .iter()
            .map(|coeff| coeff / &content)
            .collect();
        radicand_poly = Polynomial::new(primitive_coeffs, radicand_poly.var.clone());
        sqrt_scale *= sqrt_content;
    }

    if sqrt_scale.abs().is_one() {
        return None;
    }

    let radicand = radicand_poly.to_expr(ctx);
    Some((radicand, radicand_poly, sqrt_scale))
}

pub(super) fn scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, BigRational, BigRational)> {
    match ctx.get(arg).clone() {
        Expr::Neg(inner) => {
            let (radicand, denominator, sign, sqrt_scale) =
                scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation(
                    ctx, inner, var_name,
                )?;
            Some((radicand, denominator, -sign, sqrt_scale))
        }
        Expr::Mul(_, _) => {
            let (outer_scale, inner) = rational_scaled_single_factor_allow_unit(ctx, arg)?;
            if outer_scale.is_one() {
                return None;
            }
            let (radicand, denominator, sign, sqrt_scale) =
                scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation(
                    ctx, inner, var_name,
                )?;
            Some((radicand, denominator, sign, sqrt_scale * outer_scale))
        }
        Expr::Div(num, den) => {
            let (num, num_sign) = match ctx.get(num).clone() {
                Expr::Neg(inner) => (inner, -BigRational::one()),
                _ => (num, BigRational::one()),
            };
            let (den, den_sign) = match ctx.get(den).clone() {
                Expr::Neg(inner) => (inner, -BigRational::one()),
                _ => (den, BigRational::one()),
            };
            let (den, denominator_scale) =
                split_rational_scale_from_independent_symbolic_denominator_for_calculus_presentation(
                    ctx, den, var_name,
                )?;

            let (radicand, sqrt_scale) = if let Some((radicand, _, sqrt_scale)) =
                scaled_sqrt_polynomial_arg_for_calculus_presentation(ctx, num, var_name)
            {
                (radicand, sqrt_scale)
            } else {
                scaled_sqrt_argument_for_calculus_presentation(ctx, num)?
            };
            Some((
                radicand,
                den,
                num_sign * den_sign,
                sqrt_scale / denominator_scale,
            ))
        }
        _ => None,
    }
}

pub(super) fn inverse_tangent_sqrt_over_symbolic_constant_arg_for_calculus_presentation(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, BigRational, BigRational)> {
    scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation(ctx, arg, var_name)
}

fn split_rational_scale_from_independent_symbolic_denominator_for_calculus_presentation(
    ctx: &mut Context,
    denominator: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    if contains_named_var(ctx, denominator, var_name)
        || cas_ast::views::as_rational_const(ctx, denominator, 8).is_some()
    {
        return None;
    }

    if let Expr::Div(num, den) = ctx.get(denominator).clone() {
        let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if den_scale.is_zero()
            || contains_named_var(ctx, num, var_name)
            || cas_ast::views::as_rational_const(ctx, num, 8).is_some()
        {
            return None;
        }

        let (core, numerator_scale) =
            split_rational_scale_from_independent_symbolic_denominator_for_calculus_presentation(
                ctx, num, var_name,
            )
            .unwrap_or((num, BigRational::one()));
        return Some((core, numerator_scale / den_scale));
    }

    if let Some((scale, core)) = rational_scaled_single_factor_allow_unit(ctx, denominator) {
        if !scale.is_zero()
            && !contains_named_var(ctx, core, var_name)
            && cas_ast::views::as_rational_const(ctx, core, 8).is_none()
        {
            return Some((core, scale));
        }
    }

    Some((denominator, BigRational::one()))
}

pub(super) fn reciprocal_sqrt_like_arg_for_calculus_presentation(
    ctx: &Context,
    arg: ExprId,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(arg) {
        Expr::Div(num, den) => {
            let scale = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            if scale.is_zero() {
                return None;
            }
            Some((calculus_sqrt_like_radicand(ctx, *den)?, scale))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut radicand = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, arg) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 2.into()))
                    || radicand.replace(*base).is_some()
                {
                    return None;
                }
            }
            if scale.is_zero() {
                return None;
            }
            Some((radicand?, scale))
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            Some((*base, BigRational::one()))
        }
        _ => None,
    }
}

pub(super) fn scaled_sqrt_radicand_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (sign, expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (-BigRational::one(), *inner),
        _ => (BigRational::one(), expr),
    };

    if let Some(radicand) = extract_square_root_base(ctx, expr) {
        return Some((sign, radicand));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = sign;
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }

        let factor_radicand = extract_square_root_base(ctx, factor)?;
        if radicand.replace(factor_radicand).is_some() {
            return None;
        }
    }

    Some((scale, radicand?))
}

pub(super) fn shifted_sqrt_arg_radicand_and_sign(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(arg) {
        Expr::Add(left, right) => {
            if !contains_named_var(ctx, *left, var_name) {
                return signed_sqrt_arg_radicand_for_calculus(ctx, *right);
            }
            if !contains_named_var(ctx, *right, var_name) {
                return signed_sqrt_arg_radicand_for_calculus(ctx, *left);
            }
            None
        }
        Expr::Sub(left, right) => {
            if !contains_named_var(ctx, *left, var_name) {
                let (radicand, sign) = signed_sqrt_arg_radicand_for_calculus(ctx, *right)?;
                return Some((radicand, -sign));
            }
            if !contains_named_var(ctx, *right, var_name) {
                return signed_sqrt_arg_radicand_for_calculus(ctx, *left);
            }
            None
        }
        _ => None,
    }
}

fn signed_sqrt_arg_radicand_for_calculus(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (radicand, sign) = signed_sqrt_arg_radicand_for_calculus(ctx, *inner)?;
            Some((radicand, -sign))
        }
        _ => calculus_sqrt_like_radicand(ctx, expr).map(|radicand| (radicand, BigRational::one())),
    }
}

pub(super) fn supported_sqrt_shift_constant_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        let value = cas_ast::views::as_rational_const(ctx, *right, 8)?;
        let shift = -value;
        if supported_nonzero_sqrt_shift(&shift) {
            return Some((*left, shift));
        }
    }

    let terms = cas_math::expr_nary::add_leaves(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let first_constant = cas_ast::views::as_rational_const(ctx, terms[0], 8);
    let second_constant = cas_ast::views::as_rational_const(ctx, terms[1], 8);
    match (first_constant, second_constant) {
        (Some(value), None) if supported_nonzero_sqrt_shift(&value) => Some((terms[1], value)),
        (None, Some(value)) if supported_nonzero_sqrt_shift(&value) => Some((terms[0], value)),
        _ => None,
    }
}

pub(super) fn shifted_sqrt_positive_constant_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    let left_sqrt_base = extract_square_root_base(ctx, *left);
    let right_sqrt_base = extract_square_root_base(ctx, *right);

    let (radicand, shift_expr) = match (left_sqrt_base, right_sqrt_base) {
        (Some(radicand), None) => (radicand, *right),
        (None, Some(radicand)) => (radicand, *left),
        _ => return None,
    };

    let shift = cas_ast::views::as_rational_const(ctx, shift_expr, 8)?;
    shift.is_positive().then_some((radicand, shift))
}

struct SqrtTimesShiftedSqrtProductParts {
    sqrt_radicand: ExprId,
    shifted_radicand: ExprId,
    shift: BigRational,
}

pub(super) fn sqrt_times_nonzero_shifted_sqrt_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let parts = sqrt_times_shifted_sqrt_product_parts(ctx, expr)?;
    if !parts.shift.is_zero()
        && cas_math::expr_domain::exprs_equivalent(ctx, parts.sqrt_radicand, parts.shifted_radicand)
    {
        return Some((parts.sqrt_radicand, parts.shift));
    }

    None
}

fn sqrt_times_shifted_sqrt_product_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<SqrtTimesShiftedSqrtProductParts> {
    let factors: Vec<_> = cas_math::expr_nary::mul_leaves(ctx, expr)
        .into_iter()
        .collect();
    if factors.len() != 2 {
        return None;
    }

    sqrt_then_shifted_sqrt_product_parts(ctx, factors[0], factors[1])
}

fn sqrt_then_shifted_sqrt_product_parts(
    ctx: &Context,
    sqrt_factor: ExprId,
    shifted_factor: ExprId,
) -> Option<SqrtTimesShiftedSqrtProductParts> {
    let sqrt_radicand = extract_square_root_base(ctx, sqrt_factor)?;
    let (shifted_radicand, shift) = supported_sqrt_shift_constant_parts(ctx, shifted_factor)?;
    let shifted_radicand = extract_square_root_base(ctx, shifted_radicand)?;
    Some(SqrtTimesShiftedSqrtProductParts {
        sqrt_radicand,
        shifted_radicand,
        shift,
    })
}

fn supported_nonzero_sqrt_shift(value: &BigRational) -> bool {
    !value.is_zero()
}

//! Argument parsers for shifted square-root calculus presentation routes.
//!
//! These helpers recognize narrow `sqrt(radicand) + shift` and
//! `sqrt(radicand) * (sqrt(radicand) + shift)` shapes. They deliberately keep
//! shifted-root orientation and shift-sign policy separate from generic scaled
//! root argument parsing.

use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::presentation_utils::calculus_sqrt_like_radicand;

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

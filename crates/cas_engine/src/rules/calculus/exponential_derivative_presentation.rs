//! Post-calculus derivative presentation for exponential-shaped routes.
//!
//! This module owns the narrow exponential presentation shortcuts that used to
//! live inline in `calculus/mod.rs`. The caller keeps route order and domain
//! policy; this module only separates exponential-specific parsing and result
//! construction from the broader calculus dispatcher.

use super::polynomial_support::nonzero_affine_variable_derivative;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    fold_numeric_mul_constants_for_hold, nonzero_rational_parts,
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn sqrt_shifted_exp_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let (exp_expr, exp_arg, exp_scale, shift) =
        scaled_exp_plus_positive_rational_shift(ctx, radicand)?;
    if !exp_scale.is_positive() || !shift.is_positive() {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, exp_arg, var_name).ok()?;
    if arg_poly.degree() > 1 {
        return None;
    }
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.degree() != 0 {
        return None;
    }
    let derivative_coeff = derivative_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if derivative_coeff.is_zero() {
        return Some(ctx.num(0));
    }

    let coefficient = exp_scale * derivative_coeff * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, exp_expr);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff.is_one() {
        sqrt_radicand
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn scaled_exp_plus_positive_rational_shift(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, BigRational, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    scaled_exp_term_arg(ctx, *left)
        .zip(cas_ast::views::as_rational_const(ctx, *right, 8))
        .map(|((exp_expr, arg, exp_scale), shift)| (exp_expr, arg, exp_scale, shift))
        .or_else(|| {
            scaled_exp_term_arg(ctx, *right)
                .zip(cas_ast::views::as_rational_const(ctx, *left, 8))
                .map(|((exp_expr, arg, exp_scale), shift)| (exp_expr, arg, exp_scale, shift))
        })
}

fn scaled_exp_term_arg(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(arg) = exp_term_arg(ctx, expr) {
        return Some((expr, arg, BigRational::one()));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = BigRational::one();
    let mut exp_term = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        if exp_term.is_none() {
            if let Some(arg) = exp_term_arg(ctx, factor) {
                exp_term = Some((factor, arg));
                continue;
            }
        }
        return None;
    }

    let (exp_expr, arg) = exp_term?;
    Some((exp_expr, arg, scale))
}

fn exp_term_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Exp) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => Some(*exp),
        _ => None,
    }
}

fn same_arg_unary_pair_for_calculus_presentation(
    ctx: &Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    let left = cas_ast::hold::unwrap_internal_hold(ctx, left);
    let right = cas_ast::hold::unwrap_internal_hold(ctx, right);
    let Expr::Function(left_fn, left_args) = ctx.get(left) else {
        return None;
    };
    let Expr::Function(right_fn, right_args) = ctx.get(right) else {
        return None;
    };
    if left_args.len() != 1
        || right_args.len() != 1
        || ctx.builtin_of(*left_fn) != Some(left_builtin)
        || ctx.builtin_of(*right_fn) != Some(right_builtin)
        || compare_expr(ctx, left_args[0], right_args[0]) != std::cmp::Ordering::Equal
    {
        return None;
    }

    Some(left_args[0])
}

fn sin_minus_cos_arg_for_calculus_presentation(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Sub(left, right) => same_arg_unary_pair_for_calculus_presentation(
            ctx,
            left,
            BuiltinFn::Sin,
            right,
            BuiltinFn::Cos,
        ),
        Expr::Add(left, right) => {
            if let Expr::Neg(negated_right) = ctx.get(right) {
                if let Some(arg) = same_arg_unary_pair_for_calculus_presentation(
                    ctx,
                    left,
                    BuiltinFn::Sin,
                    *negated_right,
                    BuiltinFn::Cos,
                ) {
                    return Some(arg);
                }
            }
            if let Expr::Neg(negated_left) = ctx.get(left) {
                if let Some(arg) = same_arg_unary_pair_for_calculus_presentation(
                    ctx,
                    right,
                    BuiltinFn::Sin,
                    *negated_left,
                    BuiltinFn::Cos,
                ) {
                    return Some(arg);
                }
            }
            None
        }
        _ => None,
    }
}

fn sin_plus_cos_arg_for_calculus_presentation(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr).clone() else {
        return None;
    };

    same_arg_unary_pair_for_calculus_presentation(ctx, left, BuiltinFn::Sin, right, BuiltinFn::Cos)
        .or_else(|| {
            same_arg_unary_pair_for_calculus_presentation(
                ctx,
                right,
                BuiltinFn::Sin,
                left,
                BuiltinFn::Cos,
            )
        })
}

pub(super) fn exp_trig_by_parts_primitive_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut exp_expr = None;
    let mut exp_arg = None;
    let mut trig_result = None;
    for factor in factors {
        if let Some(arg) = exp_term_arg(ctx, factor) {
            if exp_expr.replace(factor).is_some() || exp_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }

        if let Some(arg) = sin_minus_cos_arg_for_calculus_presentation(ctx, factor) {
            if trig_result.replace((arg, BuiltinFn::Sin)).is_some() {
                return None;
            }
            continue;
        }

        if let Some(arg) = sin_plus_cos_arg_for_calculus_presentation(ctx, factor) {
            if trig_result.replace((arg, BuiltinFn::Cos)).is_some() {
                return None;
            }
            continue;
        }

        scale *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
    }

    let exp_expr = exp_expr?;
    let exp_arg = exp_arg?;
    let (trig_arg, result_builtin) = trig_result?;
    if compare_expr(ctx, exp_arg, trig_arg) != std::cmp::Ordering::Equal {
        return None;
    }

    let derivative_coeff = nonzero_affine_variable_derivative(ctx, exp_arg, var_name)?;
    let trig = ctx.call_builtin(result_builtin, vec![exp_arg]);
    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[exp_expr, trig]);
    let scaled = scale_expr_for_calculus_presentation(
        ctx,
        scale * BigRational::from_integer(2.into()) * derivative_coeff,
        product,
    );

    Some(fold_numeric_mul_constants_for_hold(ctx, scaled))
}

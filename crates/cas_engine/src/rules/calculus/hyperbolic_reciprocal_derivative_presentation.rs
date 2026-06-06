//! Compact derivative presentation for hyperbolic reciprocal quotients.

use super::presentation_utils::{is_calculus_presentation_one, squared_expr};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
};
use super::signed_factor_presentation::signed_mul_leaves_for_calculus_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;

#[derive(Clone, Copy)]
struct HyperbolicReciprocalQuotient {
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
    arg: ExprId,
}

pub(crate) fn constant_scaled_hyperbolic_reciprocal_derivative_quotient_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let mut scale_factors = Vec::new();
    let mut quotient_part = None;

    for factor in signed_mul_leaves_for_calculus_presentation(ctx, target) {
        if let Some(parts) = hyperbolic_reciprocal_derivative_quotient_factor(ctx, factor) {
            if quotient_part.replace(parts).is_some() {
                return None;
            }
            continue;
        }

        if contains_named_var(ctx, factor, var_name) {
            return None;
        }
        if !is_calculus_presentation_one(ctx, factor) {
            scale_factors.push(factor);
        }
    }

    let parts = quotient_part?;
    let arg_derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx, parts.arg, var_name,
    )?;
    if cas_ast::views::as_rational_const(ctx, arg_derivative, 8)
        .is_some_and(|value| value.is_zero())
        || contains_named_var(ctx, arg_derivative, var_name)
    {
        return None;
    }

    let denominator_arg = ctx.call_builtin(parts.denominator_builtin, vec![parts.arg]);
    let numerator_arg = ctx.call_builtin(parts.numerator_builtin, vec![parts.arg]);
    let denominator_square = squared_expr(ctx, denominator_arg);
    let numerator_square = squared_expr(ctx, numerator_arg);
    let two = ctx.num(2);
    let two_numerator_square =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, numerator_square]);
    let chain_core = ctx.add(Expr::Sub(denominator_square, two_numerator_square));

    scale_factors.push(arg_derivative);
    scale_factors.push(chain_core);
    let (numerator, rational_denominator) =
        split_signed_product_for_hyperbolic_reciprocal_presentation(ctx, &scale_factors)?;

    let three = ctx.num(3);
    let denominator_cube = ctx.add(Expr::Pow(denominator_arg, three));
    let denominator = if rational_denominator.is_one() {
        denominator_cube
    } else {
        let rational_denominator =
            rational_const_for_calculus_presentation(ctx, rational_denominator);
        cas_math::expr_nary::build_balanced_mul(ctx, &[rational_denominator, denominator_cube])
    };
    let result = ctx.add(Expr::Div(numerator, denominator));

    let mut required_conditions = Vec::new();
    if parts.denominator_builtin == BuiltinFn::Sinh {
        required_conditions.push(crate::ImplicitCondition::NonZero(denominator_arg));
    }

    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

fn hyperbolic_reciprocal_derivative_quotient_factor(
    ctx: &mut Context,
    factor: ExprId,
) -> Option<HyperbolicReciprocalQuotient> {
    let Expr::Div(numerator, denominator) = ctx.get(factor).clone() else {
        return None;
    };
    let (numerator_builtin, numerator_arg) = unary_hyperbolic_arg(ctx, numerator)?;
    let (denominator_builtin, denominator_arg) = hyperbolic_square_arg(ctx, denominator)?;
    if cas_ast::ordering::compare_expr(ctx, numerator_arg, denominator_arg) != Ordering::Equal {
        return None;
    }

    match (numerator_builtin, denominator_builtin) {
        (BuiltinFn::Sinh, BuiltinFn::Cosh) | (BuiltinFn::Cosh, BuiltinFn::Sinh) => {
            Some(HyperbolicReciprocalQuotient {
                numerator_builtin,
                denominator_builtin,
                arg: numerator_arg,
            })
        }
        _ => None,
    }
}

fn unary_hyperbolic_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sinh) => Some((BuiltinFn::Sinh, args[0])),
        Some(BuiltinFn::Cosh) => Some((BuiltinFn::Cosh, args[0])),
        _ => None,
    }
}

fn hyperbolic_square_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exponent) = ctx.get(expr).clone() else {
        return None;
    };
    if cas_ast::views::as_rational_const(ctx, exponent, 8)
        .is_none_or(|value| value != BigRational::from_integer(2.into()))
    {
        return None;
    }
    unary_hyperbolic_arg(ctx, base)
}

fn split_signed_product_for_hyperbolic_reciprocal_presentation(
    ctx: &mut Context,
    factors: &[ExprId],
) -> Option<(ExprId, BigRational)> {
    let mut coeff = BigRational::one();
    let mut unsigned_factors = Vec::new();

    for factor in factors {
        for leaf in signed_mul_leaves_for_calculus_presentation(ctx, *factor) {
            if let Some(value) = cas_ast::views::as_rational_const(ctx, leaf, 8) {
                coeff *= value;
                continue;
            }

            let mut absorbed_rational_denominator = false;
            if let Expr::Div(numerator, denominator) = ctx.get(leaf).clone() {
                if let Some(denominator_scale) =
                    cas_ast::views::as_rational_const(ctx, denominator, 8)
                {
                    if !denominator_scale.is_zero() {
                        coeff /= denominator_scale;
                        for numerator_leaf in
                            signed_mul_leaves_for_calculus_presentation(ctx, numerator)
                        {
                            if let Some(value) =
                                cas_ast::views::as_rational_const(ctx, numerator_leaf, 8)
                            {
                                coeff *= value;
                            } else {
                                unsigned_factors.push(numerator_leaf);
                            }
                        }
                        absorbed_rational_denominator = true;
                    }
                }
            }
            if !absorbed_rational_denominator {
                unsigned_factors.push(leaf);
            }
        }
    }

    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coeff)?;
    let numerator = if unsigned_factors.is_empty() {
        rational_const_for_calculus_presentation(ctx, numerator_coeff)
    } else if numerator_coeff.is_one() {
        cas_math::expr_nary::build_balanced_mul(ctx, &unsigned_factors)
    } else {
        let numerator_coeff = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        let mut numerator_factors = vec![numerator_coeff];
        numerator_factors.extend(unsigned_factors);
        cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors)
    };

    Some((numerator, denominator_coeff))
}

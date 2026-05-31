//! Post-calculus presentation for affine hyperbolic odd-power primitives.
//!
//! This module owns the route that recognizes derivatives of compact
//! `sinh`/`cosh` primitive families such as cubic, fifth, and seventh odd
//! powers. It intentionally mirrors the source-order behavior previously held
//! in `calculus/mod.rs`; callers keep the same priority and fallback policy.

use super::polynomial_support::nonzero_affine_variable_derivative;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::scale_expr_for_calculus_presentation;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

fn hyperbolic_power_term_for_derivative_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, u32)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    if let Expr::Function(fn_id, args) = ctx.get(expr).clone() {
        if args.len() != 1 {
            return None;
        }
        return match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Sinh | BuiltinFn::Cosh) => Some((ctx.builtin_of(fn_id)?, args[0], 1)),
            _ => None,
        };
    }

    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let power = cas_ast::views::as_rational_const(ctx, exp, 4)?;
    if power.denom() != &1.into() {
        return None;
    }
    let power = power.numer().to_u32()?;
    if !matches!(power, 3 | 5 | 7) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sinh | BuiltinFn::Cosh) => Some((ctx.builtin_of(fn_id)?, args[0], power)),
        _ => None,
    }
}

fn collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
    ctx: &mut Context,
    expr: ExprId,
    scale: BigRational,
    terms: &mut Vec<(BuiltinFn, ExprId, u32, BigRational)>,
) -> Option<()> {
    if scale.is_zero() {
        return Some(());
    }

    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Number(value) if value.is_zero() => Some(()),
        Expr::Add(left, right) => {
            collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
                ctx,
                left,
                scale.clone(),
                terms,
            )?;
            collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
                ctx, right, scale, terms,
            )
        }
        Expr::Sub(left, right) => {
            collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
                ctx,
                left,
                scale.clone(),
                terms,
            )?;
            collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
                ctx, right, -scale, terms,
            )
        }
        Expr::Neg(inner) => collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
            ctx, inner, -scale, terms,
        ),
        Expr::Div(num, den) => {
            let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
            if den_scale.is_zero() {
                return None;
            }
            collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
                ctx,
                num,
                scale / den_scale,
                terms,
            )
        }
        Expr::Mul(_, _) => {
            let mut term_scale = scale;
            let mut non_numeric = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    term_scale *= value;
                    continue;
                }
                if non_numeric.replace(factor).is_some() {
                    return None;
                }
            }
            collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
                ctx,
                non_numeric?,
                term_scale,
                terms,
            )
        }
        _ => {
            let (builtin, arg, power) =
                hyperbolic_power_term_for_derivative_presentation(ctx, expr)?;
            terms.push((builtin, arg, power, scale));
            Some(())
        }
    }
}

fn hyperbolic_derivative_presentation_coeff(
    terms: &[(BuiltinFn, ExprId, u32, BigRational)],
    power: u32,
) -> BigRational {
    terms
        .iter()
        .filter_map(|(_, _, term_power, coeff)| (*term_power == power).then_some(coeff.clone()))
        .fold(BigRational::zero(), |acc, coeff| acc + coeff)
}

fn affine_hyperbolic_cubic_primitive_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
        ctx,
        target,
        BigRational::one(),
        &mut terms,
    )?;
    if terms.len() != 2 {
        return None;
    }

    let (builtin, arg) = terms
        .iter()
        .find_map(|(builtin, arg, _, coeff)| (!coeff.is_zero()).then_some((*builtin, *arg)))?;
    if terms.iter().any(|(term_builtin, term_arg, power, coeff)| {
        !coeff.is_zero()
            && (*term_builtin != builtin
                || compare_expr(ctx, *term_arg, arg) != std::cmp::Ordering::Equal
                || !matches!(*power, 1 | 3))
    }) {
        return None;
    }

    let linear_coeff = hyperbolic_derivative_presentation_coeff(&terms, 1);
    let cubic_coeff = hyperbolic_derivative_presentation_coeff(&terms, 3);
    if linear_coeff.is_zero() || cubic_coeff.is_zero() {
        return None;
    }

    let one_third = BigRational::new(1.into(), 3.into());
    let (companion_builtin, primitive_scale) = match builtin {
        BuiltinFn::Cosh if cubic_coeff == -linear_coeff.clone() * one_third.clone() => {
            (BuiltinFn::Sinh, -linear_coeff)
        }
        BuiltinFn::Sinh if cubic_coeff == linear_coeff.clone() * one_third => {
            (BuiltinFn::Cosh, linear_coeff)
        }
        _ => return None,
    };

    let arg_derivative = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let derivative_coeff = primitive_scale * arg_derivative;
    if derivative_coeff.is_zero() {
        return None;
    }

    let companion = ctx.call_builtin(companion_builtin, vec![arg]);
    let three = ctx.num(3);
    let companion_cubed = ctx.add(Expr::Pow(companion, three));
    Some(scale_expr_for_calculus_presentation(
        ctx,
        derivative_coeff,
        companion_cubed,
    ))
}

fn affine_hyperbolic_fifth_primitive_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
        ctx,
        target,
        BigRational::one(),
        &mut terms,
    )?;
    if terms.len() != 3 {
        return None;
    }

    let (builtin, arg) = terms
        .iter()
        .find_map(|(builtin, arg, _, coeff)| (!coeff.is_zero()).then_some((*builtin, *arg)))?;
    if terms.iter().any(|(term_builtin, term_arg, power, coeff)| {
        !coeff.is_zero()
            && (*term_builtin != builtin
                || compare_expr(ctx, *term_arg, arg) != std::cmp::Ordering::Equal
                || !matches!(*power, 1 | 3 | 5))
    }) {
        return None;
    }

    let linear_coeff = hyperbolic_derivative_presentation_coeff(&terms, 1);
    let cubic_coeff = hyperbolic_derivative_presentation_coeff(&terms, 3);
    let fifth_coeff = hyperbolic_derivative_presentation_coeff(&terms, 5);
    if linear_coeff.is_zero() || cubic_coeff.is_zero() || fifth_coeff.is_zero() {
        return None;
    }

    let two_thirds = BigRational::new(2.into(), 3.into());
    let one_fifth = BigRational::new(1.into(), 5.into());
    let expected_fifth = linear_coeff.clone() * one_fifth;
    let (companion_builtin, primitive_scale) = match builtin {
        BuiltinFn::Cosh
            if cubic_coeff == -(linear_coeff.clone() * two_thirds.clone())
                && fifth_coeff == expected_fifth =>
        {
            (BuiltinFn::Sinh, linear_coeff)
        }
        BuiltinFn::Sinh
            if cubic_coeff == linear_coeff.clone() * two_thirds
                && fifth_coeff == expected_fifth =>
        {
            (BuiltinFn::Cosh, linear_coeff)
        }
        _ => return None,
    };

    let arg_derivative = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let derivative_coeff = primitive_scale * arg_derivative;
    if derivative_coeff.is_zero() {
        return None;
    }

    let companion = ctx.call_builtin(companion_builtin, vec![arg]);
    let five = ctx.num(5);
    let companion_fifth = ctx.add(Expr::Pow(companion, five));
    Some(scale_expr_for_calculus_presentation(
        ctx,
        derivative_coeff,
        companion_fifth,
    ))
}

fn affine_hyperbolic_seventh_primitive_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_hyperbolic_power_terms_for_derivative_presentation(
        ctx,
        target,
        BigRational::one(),
        &mut terms,
    )?;
    if terms.len() != 4 {
        return None;
    }

    let (builtin, arg) = terms
        .iter()
        .find_map(|(builtin, arg, _, coeff)| (!coeff.is_zero()).then_some((*builtin, *arg)))?;
    if terms.iter().any(|(term_builtin, term_arg, power, coeff)| {
        !coeff.is_zero()
            && (*term_builtin != builtin
                || compare_expr(ctx, *term_arg, arg) != std::cmp::Ordering::Equal
                || !matches!(*power, 1 | 3 | 5 | 7))
    }) {
        return None;
    }

    let linear_coeff = hyperbolic_derivative_presentation_coeff(&terms, 1);
    let cubic_coeff = hyperbolic_derivative_presentation_coeff(&terms, 3);
    let fifth_coeff = hyperbolic_derivative_presentation_coeff(&terms, 5);
    let seventh_coeff = hyperbolic_derivative_presentation_coeff(&terms, 7);
    if linear_coeff.is_zero()
        || cubic_coeff.is_zero()
        || fifth_coeff.is_zero()
        || seventh_coeff.is_zero()
    {
        return None;
    }

    let three_fifths = BigRational::new(3.into(), 5.into());
    let one_seventh = BigRational::new(1.into(), 7.into());
    let (companion_builtin, primitive_scale) = match builtin {
        BuiltinFn::Cosh
            if cubic_coeff == -linear_coeff.clone()
                && fifth_coeff == linear_coeff.clone() * three_fifths.clone()
                && seventh_coeff == -(linear_coeff.clone() * one_seventh.clone()) =>
        {
            (BuiltinFn::Sinh, -linear_coeff)
        }
        BuiltinFn::Sinh
            if cubic_coeff == linear_coeff.clone()
                && fifth_coeff == linear_coeff.clone() * three_fifths
                && seventh_coeff == linear_coeff.clone() * one_seventh =>
        {
            (BuiltinFn::Cosh, linear_coeff)
        }
        _ => return None,
    };

    let arg_derivative = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let derivative_coeff = primitive_scale * arg_derivative;
    if derivative_coeff.is_zero() {
        return None;
    }

    let companion = ctx.call_builtin(companion_builtin, vec![arg]);
    let seven = ctx.num(7);
    let companion_seventh = ctx.add(Expr::Pow(companion, seven));
    Some(scale_expr_for_calculus_presentation(
        ctx,
        derivative_coeff,
        companion_seventh,
    ))
}

pub(crate) fn affine_hyperbolic_odd_primitive_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(result) =
        affine_hyperbolic_cubic_primitive_derivative_presentation(ctx, target, var_name)
    {
        return Some(result);
    }
    if let Some(result) =
        affine_hyperbolic_fifth_primitive_derivative_presentation(ctx, target, var_name)
    {
        return Some(result);
    }
    affine_hyperbolic_seventh_primitive_derivative_presentation(ctx, target, var_name)
}

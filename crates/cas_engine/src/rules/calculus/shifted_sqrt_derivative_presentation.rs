//! Compact derivative presentation for shifted-square-root quotient routes.
//!
//! This module owns the `sqrt(radicand) + shift` quotient/product family used
//! by differentiation shortcuts and post-calculus presentation. It deliberately
//! preserves the existing route order from `calculus/mod.rs` and keeps
//! domain-condition construction tied to the shifted-root policy.

use super::differentiation::differentiate;
use super::domain_checks::shifted_sqrt_product_required_conditions;
use super::polynomial_support::split_polynomial_content_for_calculus_presentation;
use super::presentation_utils::squared_expr;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use super::scaled_sqrt_args::scaled_square_root_radicand_for_calculus_presentation;
use super::shifted_sqrt_args::{
    shifted_sqrt_positive_constant_parts, sqrt_times_nonzero_shifted_sqrt_parts,
};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn sqrt_over_positive_shifted_sqrt_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let (numerator_scale, numerator_radicand) =
        scaled_square_root_radicand_for_calculus_presentation(ctx, numerator_expr)?;
    let (denominator_radicand, shift) =
        shifted_sqrt_positive_constant_parts(ctx, denominator_expr)?;
    if compare_expr(ctx, numerator_radicand, denominator_radicand) != std::cmp::Ordering::Equal {
        return None;
    }

    let d_radicand = differentiate(ctx, numerator_radicand, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_radicand, 8).is_some_and(|value| value.is_zero()) {
        return Some((ctx.num(0), numerator_radicand));
    }

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_radicand);
    let coefficient =
        numerator_scale * shift.clone() * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![numerator_radicand]);
    let shift_expr = rational_const_for_calculus_presentation(ctx, shift);
    let shifted_sqrt = ctx.add(Expr::Add(sqrt_radicand, shift_expr));
    let shifted_sqrt_squared = squared_expr(ctx, shifted_sqrt);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt_squared]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some((
        ctx.add(Expr::Div(numerator, denominator)),
        numerator_radicand,
    ))
}

pub(crate) fn sqrt_over_positive_shifted_sqrt_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (result, required_positive) =
        sqrt_over_positive_shifted_sqrt_derivative(ctx, target, var_name)?;
    Some((
        result,
        vec![crate::ImplicitCondition::Positive(required_positive)],
    ))
}

pub(super) fn reciprocal_positive_shifted_sqrt_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    if numerator_scale.is_zero() {
        return Some(ctx.num(0));
    }

    let (radicand, shift) = shifted_sqrt_positive_constant_parts(ctx, den)?;
    let d_radicand = differentiate(ctx, radicand, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_radicand, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let shift_expr = rational_const_for_calculus_presentation(ctx, shift);
    let shifted_sqrt = ctx.add(Expr::Add(sqrt_radicand, shift_expr));
    let shifted_sqrt_squared = squared_expr(ctx, shifted_sqrt);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt_squared]);

    if let Some(d_radicand_scale) = cas_ast::views::as_rational_const(ctx, d_radicand, 8) {
        let coefficient = -numerator_scale * d_radicand_scale / BigRational::from_integer(2.into());
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        let denominator = if denominator_coeff == BigRational::one() {
            core_denominator
        } else {
            let denominator_scale =
                rational_const_for_calculus_presentation(ctx, denominator_coeff);
            cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
        };

        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let negative_scale = ctx.add(Expr::Number(-numerator_scale));
    let numerator = ctx.add(Expr::Mul(negative_scale, d_radicand));
    let two = ctx.num(2);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[two, core_denominator]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    if numerator_scale.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let (radicand, shift) = sqrt_times_nonzero_shifted_sqrt_parts(ctx, den)?;
    let d_radicand = differentiate(ctx, radicand, var_name)?;
    let d_radicand_scale = cas_ast::views::as_rational_const(ctx, d_radicand, 8)?;
    if d_radicand_scale.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let two = ctx.num(2);
    let two_sqrt = ctx.add(Expr::Mul(two, sqrt_radicand));
    let shift_expr = rational_const_for_calculus_presentation(ctx, shift.clone());
    let numerator_core = ctx.add(Expr::Add(two_sqrt, shift_expr));
    let coefficient = -numerator_scale * d_radicand_scale;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let shifted_sqrt = ctx.add(Expr::Add(sqrt_radicand, shift_expr));
    let shifted_sqrt_squared = squared_expr(ctx, shifted_sqrt);
    let mut denominator_parts = Vec::new();
    let denominator_scale = denominator_coeff * BigRational::from_integer(2.into());
    if denominator_scale != BigRational::one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_scale,
        ));
    }
    denominator_parts.push(radicand);
    denominator_parts.push(sqrt_radicand);
    denominator_parts.push(shifted_sqrt_squared);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);

    let result = ctx.add(Expr::Div(numerator, denominator));
    let required_conditions =
        shifted_sqrt_product_required_conditions(radicand, &shift, shifted_sqrt);

    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

pub(super) fn inverse_tangent_reciprocal_sqrt_shifted_sqrt_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => -BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => BigRational::one(),
        _ => return None,
    };

    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(args[0]).clone() else {
        return None;
    };
    let argument_scale = cas_ast::views::as_rational_const(ctx, numerator_expr, 8)?;
    if argument_scale.is_zero() {
        return None;
    }

    let (radicand, shift) = sqrt_times_nonzero_shifted_sqrt_parts(ctx, denominator_expr)?;
    let d_radicand = differentiate(ctx, radicand, var_name)?;
    let d_radicand_scale = cas_ast::views::as_rational_const(ctx, d_radicand, 8)?;
    if d_radicand_scale.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let two = ctx.num(2);
    let two_sqrt = ctx.add(Expr::Mul(two, sqrt_radicand));
    let shift_expr = rational_const_for_calculus_presentation(ctx, shift.clone());
    let numerator_core = ctx.add(Expr::Add(two_sqrt, shift_expr));
    let coefficient = derivative_sign * argument_scale.clone() * d_radicand_scale;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let shifted_sqrt = ctx.add(Expr::Add(sqrt_radicand, shift_expr));
    let shifted_sqrt_squared = squared_expr(ctx, shifted_sqrt);
    let root_term = cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, shifted_sqrt_squared]);
    let scale_square = argument_scale.clone() * argument_scale;
    let scale_square_expr = rational_const_for_calculus_presentation(ctx, scale_square);
    let gap = ctx.add(Expr::Add(root_term, scale_square_expr));

    let mut denominator_parts = Vec::new();
    let denominator_scale = denominator_coeff * BigRational::from_integer(2.into());
    if denominator_scale != BigRational::one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_scale,
        ));
    }
    denominator_parts.push(sqrt_radicand);
    denominator_parts.push(gap);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);

    let compact = ctx.add(Expr::Div(numerator, denominator));
    let required_conditions =
        shifted_sqrt_product_required_conditions(radicand, &shift, shifted_sqrt);

    Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions))
}

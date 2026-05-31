//! Compact derivative presentation for log/square-root quotient routes.
//!
//! This module owns the post-calculus presentation family for
//! `ln(q(x)) / sqrt(q(x))` and `sqrt(q(x)) / ln(q(x))`. It keeps the
//! parameter-scale parser local because the current policy is only proven for
//! these two reciprocal log/root orientations.

use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::{
    calculus_sqrt_like_radicand, is_calculus_presentation_one, structurally_equivalent_for_calculus,
};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn log_over_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (outer_scale, outer_parameter_scale, target) =
        parameter_scaled_single_core_for_calculus_presentation(ctx, target, var_name)?;
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let numerator_expr = cas_ast::hold::unwrap_internal_hold(ctx, numerator_expr);
    let denominator_expr = cas_ast::hold::unwrap_internal_hold(ctx, denominator_expr);
    let (numerator_scale, numerator_parameter_scale, numerator_expr) =
        parameter_scaled_single_core_for_calculus_presentation(ctx, numerator_expr, var_name)?;
    let (denominator_scale, denominator_parameter_scale, denominator_expr) =
        parameter_scaled_single_core_for_calculus_presentation(ctx, denominator_expr, var_name)?;
    if denominator_scale.is_zero() {
        return None;
    }
    let presentation_scale = outer_scale * numerator_scale / denominator_scale;
    let numerator_expr = cas_ast::hold::unwrap_internal_hold(ctx, numerator_expr);
    let Expr::Function(fn_id, args) = ctx.get(numerator_expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let radicand = calculus_sqrt_like_radicand(ctx, denominator_expr)?;
    if !structurally_equivalent_for_calculus(ctx, args[0], radicand) {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    if derivative_poly.degree() > 2 {
        return None;
    }

    let raw_derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        if let Some(value) = cas_ast::views::as_rational_const(ctx, raw_derivative, 8) {
            (ctx.num(1), value)
        } else {
            split_polynomial_content_for_calculus_presentation(ctx, raw_derivative)
        };

    let two = ctx.num(2);
    let ln_radicand = ctx.call_builtin(BuiltinFn::Ln, vec![radicand]);
    let log_gap = ctx.add(Expr::Sub(two, ln_radicand));
    let mut numerator_core_factors = Vec::new();
    if let Some(scale) = outer_parameter_scale {
        numerator_core_factors.push(scale);
    }
    if let Some(scale) = numerator_parameter_scale {
        numerator_core_factors.push(scale);
    }
    if !is_calculus_presentation_one(ctx, derivative_core) {
        numerator_core_factors.push(derivative_core);
    }
    numerator_core_factors.push(log_gap);
    let numerator_core = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_core_factors);

    let coefficient = presentation_scale * derivative_content / BigRational::from_integer(2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_core_factors = Vec::new();
    if let Some(scale) = denominator_parameter_scale {
        denominator_core_factors.push(scale);
    }
    denominator_core_factors.push(radicand);
    denominator_core_factors.push(sqrt_radicand);
    let denominator_core = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_core_factors);
    let denominator = if denominator_coeff == BigRational::one() {
        denominator_core
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, denominator_core])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn sqrt_over_log_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (outer_scale, outer_parameter_scale, target) =
        parameter_scaled_single_core_for_calculus_presentation(ctx, target, var_name)?;
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let numerator_expr = cas_ast::hold::unwrap_internal_hold(ctx, numerator_expr);
    let denominator_expr = cas_ast::hold::unwrap_internal_hold(ctx, denominator_expr);
    let (numerator_scale, numerator_parameter_scale, numerator_expr) =
        parameter_scaled_single_core_for_calculus_presentation(ctx, numerator_expr, var_name)?;
    let (denominator_scale, denominator_parameter_scale, denominator_expr) =
        parameter_scaled_single_core_for_calculus_presentation(ctx, denominator_expr, var_name)?;
    if denominator_scale.is_zero() {
        return None;
    }
    denominator_parameter_scale?;

    let radicand = calculus_sqrt_like_radicand(ctx, numerator_expr)?;
    let Expr::Function(fn_id, args) = ctx.get(denominator_expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln)
        || args.len() != 1
        || !structurally_equivalent_for_calculus(ctx, args[0], radicand)
    {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    if derivative_poly.degree() > 2 {
        return None;
    }

    let raw_derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        if let Some(value) = cas_ast::views::as_rational_const(ctx, raw_derivative, 8) {
            (ctx.num(1), value)
        } else {
            split_polynomial_content_for_calculus_presentation(ctx, raw_derivative)
        };

    let two = ctx.num(2);
    let ln_radicand = ctx.call_builtin(BuiltinFn::Ln, vec![radicand]);
    let log_gap = ctx.add(Expr::Sub(ln_radicand, two));
    let mut numerator_core_factors = Vec::new();
    if let Some(scale) = outer_parameter_scale {
        numerator_core_factors.push(scale);
    }
    if let Some(scale) = numerator_parameter_scale {
        numerator_core_factors.push(scale);
    }
    if !is_calculus_presentation_one(ctx, derivative_core) {
        numerator_core_factors.push(derivative_core);
    }
    numerator_core_factors.push(log_gap);
    let numerator_core = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_core_factors);

    let presentation_scale = outer_scale * numerator_scale / denominator_scale;
    let coefficient = presentation_scale * derivative_content / BigRational::from_integer(2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let ln_squared = ctx.add(Expr::Pow(ln_radicand, two));
    let mut denominator_core_factors = Vec::new();
    if let Some(scale) = denominator_parameter_scale {
        denominator_core_factors.push(scale);
    }
    denominator_core_factors.push(ln_squared);
    denominator_core_factors.push(sqrt_radicand);
    let denominator_core = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_core_factors);
    let denominator = if denominator_coeff == BigRational::one() {
        denominator_core
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, denominator_core])
    };

    let result = ctx.add(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, result))
}

fn parameter_scaled_single_core_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, Option<ExprId>, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, parameter_scale, core) =
            parameter_scaled_single_core_for_calculus_presentation(ctx, inner, var_name)?;
        return Some((-scale, parameter_scale, core));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut parameter_scale = None;
    let mut core = None;

    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if is_independent_parameter_scale_for_calculus_presentation(ctx, factor, var_name) {
            if parameter_scale.replace(factor).is_some() {
                return None;
            }
        } else if core.replace(factor).is_some() {
            return None;
        }
    }

    Some((scale, parameter_scale, core?))
}

fn is_independent_parameter_scale_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) != var_name)
}

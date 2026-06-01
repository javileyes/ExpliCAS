use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

use super::differentiation::differentiate;
use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_compaction::{
    bounded_sin_cos_term_bound_for_calculus_presentation,
    compact_numeric_mul_factors_for_calculus_presentation,
    scale_ordered_product_for_calculus_presentation,
    split_signed_numeric_scale_single_core_for_calculus_presentation,
    unary_variable_builtin_arg_for_calculus_presentation,
};
use super::presentation_utils::calculus_sqrt_like_radicand;
use super::scalar_presentation::scale_expr_for_calculus_presentation;

fn tan_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Tan) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn scaled_tan_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = tan_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Cot) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

fn scaled_cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = cot_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn scaled_sin_over_cos_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let sin_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, num, var_name, BuiltinFn::Sin)?;
    let cos_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, den, var_name, BuiltinFn::Cos)?;
    (sin_arg == cos_arg).then_some((scale, sin_arg))
}

fn scaled_cos_over_sin_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let cos_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, num, var_name, BuiltinFn::Cos)?;
    let sin_arg =
        unary_variable_builtin_arg_for_calculus_presentation(ctx, den, var_name, BuiltinFn::Sin)?;
    (cos_arg == sin_arg).then_some((scale, cos_arg))
}

pub(super) fn scaled_tan_or_cot_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, BuiltinFn)> {
    if let Some((scale, arg)) =
        scaled_tan_variable_arg_for_calculus_presentation(ctx, expr, var_name).or_else(|| {
            scaled_sin_over_cos_variable_arg_for_calculus_presentation(ctx, expr, var_name)
        })
    {
        return Some((scale, arg, BuiltinFn::Cos));
    }

    scaled_cot_variable_arg_for_calculus_presentation(ctx, expr, var_name)
        .or_else(|| scaled_cos_over_sin_variable_arg_for_calculus_presentation(ctx, expr, var_name))
        .map(|(scale, arg)| (-scale, arg, BuiltinFn::Sin))
}

pub(super) fn scaled_sec_or_csc_variable_derivative_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let core = cas_ast::hold::unwrap_internal_hold(ctx, core);
    let Expr::Function(fn_id, args) = ctx.get(core).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    if ctx.sym_name(*sym_id) != var_name {
        return None;
    }

    match ctx.builtin_of(fn_id)? {
        BuiltinFn::Sec => {
            let sec = ctx.call_builtin(BuiltinFn::Sec, vec![args[0]]);
            let tan = ctx.call_builtin(BuiltinFn::Tan, vec![args[0]]);
            let derivative = cas_math::expr_nary::build_balanced_mul(ctx, &[sec, tan]);
            let derivative = scale_expr_for_calculus_presentation(ctx, scale, derivative);
            let cos = ctx.call_builtin(BuiltinFn::Cos, vec![args[0]]);
            Some((derivative, crate::ImplicitCondition::NonZero(cos)))
        }
        BuiltinFn::Csc => {
            let csc = ctx.call_builtin(BuiltinFn::Csc, vec![args[0]]);
            let cot = ctx.call_builtin(BuiltinFn::Cot, vec![args[0]]);
            let derivative = scale_ordered_product_for_calculus_presentation(ctx, -scale, csc, cot);
            let sin = ctx.call_builtin(BuiltinFn::Sin, vec![args[0]]);
            Some((derivative, crate::ImplicitCondition::NonZero(sin)))
        }
        _ => None,
    }
}

fn ln_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) {
        return None;
    }
    let Expr::Variable(sym_id) = ctx.get(args[0]) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(args[0])
}

pub(super) fn scaled_ln_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let arg = ln_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, arg))
}

fn exp_linear_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 || ctx.builtin_of(fn_id) != Some(BuiltinFn::Exp) {
                return None;
            }
            let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, args[0], var_name)?;
            if arg_poly.degree() != 1 {
                return None;
            }
            let slope = arg_poly.coeffs.get(1)?.clone();
            (!slope.is_zero()).then_some((slope, expr))
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, exp, var_name)?;
            if arg_poly.degree() != 1 {
                return None;
            }
            let slope = arg_poly.coeffs.get(1)?.clone();
            (!slope.is_zero()).then_some((slope, expr))
        }
        _ => None,
    }
}

pub(super) fn scaled_exp_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let (chain_scale, exp) = exp_linear_term_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale * chain_scale, exp))
}

fn exp_bounded_chain_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let inner = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Exp) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => exp,
        _ => return None,
    };

    bounded_sin_cos_term_bound_for_calculus_presentation(ctx, inner)?;
    let inner_derivative = differentiate(ctx, inner, var_name)?;
    if cas_ast::views::as_rational_const(ctx, inner_derivative, 8)
        .is_some_and(|value| value.is_zero())
    {
        return None;
    }
    Some((inner_derivative, expr))
}

pub(super) fn scaled_exp_bounded_chain_derivative_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let (inner_derivative, exp_term) =
        exp_bounded_chain_term_for_calculus_presentation(ctx, core, var_name)?;
    let derivative = cas_math::expr_nary::build_balanced_mul(ctx, &[inner_derivative, exp_term]);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    Some(scale_expr_for_calculus_presentation(ctx, scale, derivative))
}

fn sqrt_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let radicand = calculus_sqrt_like_radicand(ctx, expr)?;
    let Expr::Variable(sym_id) = ctx.get(radicand) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some(radicand)
}

pub(super) fn scaled_sqrt_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    let radicand = sqrt_variable_arg_for_calculus_presentation(ctx, core, var_name)?;
    Some((scale, radicand))
}

fn reciprocal_sqrt_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Div(num, den)
            if cas_ast::views::as_rational_const(ctx, num, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            sqrt_variable_arg_for_calculus_presentation(ctx, den, var_name)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            let Expr::Variable(sym_id) = ctx.get(base) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(base)
        }
        _ => None,
    }
}

pub(super) fn scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, radicand) =
            scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(ctx, inner, var_name)?;
        return Some((-scale, radicand));
    }

    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    if let Some(radicand) =
        reciprocal_sqrt_variable_arg_for_calculus_presentation(ctx, core, var_name)
    {
        return Some((scale, radicand));
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    let radicand = sqrt_variable_arg_for_calculus_presentation(ctx, den, var_name)?;
    Some((numerator_scale, radicand))
}

fn reciprocal_variable_arg_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Div(num, den)
            if cas_ast::views::as_rational_const(ctx, num, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            let Expr::Variable(sym_id) = ctx.get(den) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(den)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            let Expr::Variable(sym_id) = ctx.get(base) else {
                return None;
            };
            (ctx.sym_name(*sym_id) == var_name).then_some(base)
        }
        _ => None,
    }
}

pub(super) fn scaled_reciprocal_variable_term_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, arg) =
            scaled_reciprocal_variable_term_for_calculus_presentation(ctx, inner, var_name)?;
        return Some((-scale, arg));
    }

    let (scale, core) = split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, expr)
        .unwrap_or((BigRational::one(), expr));
    if let Some(arg) = reciprocal_variable_arg_for_calculus_presentation(ctx, core, var_name) {
        return Some((scale, arg));
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator_scale = cas_ast::views::as_rational_const(ctx, num, 8)?;
    let Expr::Variable(sym_id) = ctx.get(den) else {
        return None;
    };
    (ctx.sym_name(*sym_id) == var_name).then_some((numerator_scale, den))
}

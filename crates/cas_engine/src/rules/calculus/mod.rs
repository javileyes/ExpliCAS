//! Calculus rules: differentiation, integration, summation, and products.
//!
//! This module is split into submodules:
//! - `differentiation`: symbolic derivative computation
//! - `integration`: symbolic integral computation + helpers
//! - `summation`: finite sum/product evaluation (SumRule, ProductRule)

mod differentiation;
mod integration;
mod summation;

use crate::define_rule;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{
    render_diff_desc_with, render_integrate_desc_with, try_extract_diff_call,
    try_extract_integrate_call,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed};

use differentiation::differentiate;
use integration::{
    integrate, integrate_required_nonzero_conditions, integrate_required_positive_conditions,
};

fn atanh_diff_required_conditions(
    ctx: &mut Context,
    target: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return vec![],
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
        return vec![];
    }

    let arg = args[0];
    let open_interval = atanh_open_interval_condition(ctx, arg);
    vec![crate::ImplicitCondition::Positive(open_interval)]
}

fn squared_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(expr, two))
}

fn atanh_arg_over_sqrt_parts(ctx: &mut Context, arg: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(arg) {
        Expr::Div(num, den) => {
            let radicand = extract_square_root_base(ctx, *den)?;
            Some((*num, radicand))
        }
        Expr::Mul(_, _) => rationalized_arg_over_sqrt_parts(ctx, arg),
        _ => None,
    }
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

fn atanh_open_interval_condition(ctx: &mut Context, arg: ExprId) -> ExprId {
    if let Some((num, radicand)) = atanh_arg_over_sqrt_parts(ctx, arg) {
        let num_square = squared_expr(ctx, num);
        return ctx.add(Expr::Sub(radicand, num_square));
    }

    let one = ctx.num(1);
    let arg_sq = squared_expr(ctx, arg);
    ctx.add(Expr::Sub(one, arg_sq))
}

fn primitive_positive_gap(ctx: &mut Context, gap: ExprId) -> (ExprId, BigRational) {
    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let Ok(poly) = multipoly_from_expr(ctx, gap, &budget) else {
        return (gap, BigRational::one());
    };
    let (content, primitive) = poly.primitive_part();
    if !content.is_positive() {
        return (gap, BigRational::one());
    }
    let primitive_expr = multipoly_to_expr(&primitive, ctx);
    if content.is_one() {
        return (primitive_expr, BigRational::one());
    }

    (primitive_expr, content)
}

fn positive_integer_power_shape(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(_, exp) = ctx.get(expr) else {
        return false;
    };

    matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.is_positive())
}

fn sqrt_positive_rational_factor(ctx: &mut Context, value: BigRational) -> Option<ExprId> {
    if value.is_one() {
        return None;
    }

    let value = ctx.add(Expr::Number(value));
    Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![value]))
}

fn reciprocal_positive_rational(value: &BigRational) -> BigRational {
    BigRational::new(value.denom().clone(), value.numer().clone())
}

fn negative_half_power(ctx: &mut Context, base: ExprId) -> ExprId {
    let minus_one = ctx.num(-1);
    let two = ctx.num(2);
    let neg_half = ctx.add(Expr::Div(minus_one, two));
    ctx.add(Expr::Pow(base, neg_half))
}

fn bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    let sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => 1,
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -1,
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let d_num = differentiate(ctx, num, var_name)?;
    let numerator = if sign < 0 {
        ctx.add(Expr::Neg(d_num))
    } else {
        d_num
    };
    let gap = atanh_open_interval_condition(ctx, args[0]);
    let (gap, content) = primitive_positive_gap(ctx, gap);
    let reciprocal_content = reciprocal_positive_rational(&content);
    let numerator = if let Some(scale) = sqrt_positive_rational_factor(ctx, reciprocal_content) {
        ctx.add(Expr::Mul(scale, numerator))
    } else {
        numerator
    };
    let reciprocal_root = negative_half_power(ctx, gap);

    Some(ctx.add(Expr::Mul(numerator, reciprocal_root)))
}

fn asinh_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Asinh) || args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let d_num = differentiate(ctx, num, var_name)?;
    let num_square = squared_expr(ctx, num);
    let positive_gap = ctx.add(Expr::Add(num_square, radicand));
    let (positive_gap, content) = primitive_positive_gap(ctx, positive_gap);
    let reciprocal_content = reciprocal_positive_rational(&content);
    let d_num = if let Some(scale) = sqrt_positive_rational_factor(ctx, reciprocal_content) {
        ctx.add(Expr::Mul(scale, d_num))
    } else {
        d_num
    };
    let reciprocal_root = negative_half_power(ctx, positive_gap);

    Some(ctx.add(Expr::Mul(d_num, reciprocal_root)))
}

fn collect_atanh_open_interval_conditions(ctx: &mut Context, root: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr).clone() {
            Expr::Function(fn_id, args) => {
                if ctx.builtin_of(fn_id) == Some(BuiltinFn::Atanh) && args.len() == 1 {
                    let arg = args[0];
                    out.push(atanh_open_interval_condition(ctx, arg));
                }
                stack.extend(args);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                stack.push(l);
                stack.push(r);
            }
            Expr::Pow(base, exp) => {
                stack.push(base);
                stack.push(exp);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(inner),
            Expr::Matrix { data, .. } => stack.extend(data),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    out
}

fn atanh_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let d_num = differentiate(ctx, num, var_name)?;
    let denominator = atanh_open_interval_condition(ctx, args[0]);
    let (denominator, content) = if positive_integer_power_shape(ctx, num) {
        (denominator, BigRational::one())
    } else {
        primitive_positive_gap(ctx, denominator)
    };
    let content_squared = &content * &content;
    let adjusted_radicand = radicand_value / content_squared;
    let sqrt_radicand = ctx.add(Expr::Number(adjusted_radicand));
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![sqrt_radicand]);
    let numerator = ctx.add(Expr::Mul(sqrt_radicand, d_num));

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    let call = try_extract_integrate_call(ctx, expr)?;
    let required_nonzero = integrate_required_nonzero_conditions(ctx, call.target, &call.var_name);
    let mut required_positive =
        integrate_required_positive_conditions(ctx, call.target, &call.var_name);
    let result = integrate(ctx, call.target, &call.var_name)?;
    if required_positive.is_empty() {
        required_positive.extend(collect_atanh_open_interval_conditions(ctx, result));
    }
    let desc = render_integrate_desc_with(&call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    let required_conditions = required_nonzero
        .into_iter()
        .map(crate::ImplicitCondition::NonZero)
        .chain(
            required_positive
                .into_iter()
                .map(crate::ImplicitCondition::Positive),
        );
    Some(
        Rewrite::new(result)
            .desc(desc)
            .requires_all(required_conditions),
    )
});

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    let call = try_extract_diff_call(ctx, expr)?;
    let result =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, call.target, &call.var_name)
            .or_else(|| asinh_surd_quotient_compact_derivative(ctx, call.target, &call.var_name))
            .or_else(|| atanh_surd_quotient_compact_derivative(ctx, call.target, &call.var_name))
            .or_else(|| differentiate(ctx, call.target, &call.var_name))?;
    let required_conditions = atanh_diff_required_conditions(ctx, call.target);
    let desc = render_diff_desc_with(&call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Some(
        Rewrite::new(result)
            .desc(desc)
            .requires_all(required_conditions),
    )
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(summation::SumRule));
    simplifier.add_rule(Box::new(summation::ProductRule));
}

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
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

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

fn unwrap_internal_hold_for_calculus(ctx: &mut Context, target: ExprId) -> ExprId {
    cas_ast::hold::strip_all_holds(ctx, target)
}

fn variable_named(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

fn is_half_power_exponent(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, expr, 8)
        .is_some_and(|value| value == BigRational::new(1.into(), 2.into()))
}

fn positive_scaled_variable_factor(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if variable_named(ctx, target, var_name) {
        return Some(BigRational::one());
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut saw_variable = false;
    for factor in factors {
        if variable_named(ctx, factor, var_name) {
            if saw_variable {
                return None;
            }
            saw_variable = true;
            continue;
        }

        let value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        if !value.is_positive() {
            return None;
        }
        scale *= value;
    }

    saw_variable.then_some(scale)
}

fn nonzero_affine_variable_derivative(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 1,
        max_pow_exp: 1,
    };
    let poly = multipoly_from_expr(ctx, target, &budget).ok()?;
    if poly.vars.len() != 1 || poly.vars[0] != var_name || poly.total_degree() > 1 {
        return None;
    }

    let mut linear_coeff = BigRational::zero();
    for (coeff, mono) in &poly.terms {
        match mono.as_slice() {
            [0] => {}
            [1] => linear_coeff += coeff.clone(),
            _ => return None,
        }
    }

    (!linear_coeff.is_zero()).then_some(linear_coeff)
}

fn arctan_sqrt_scaled_variable_arg(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }

    let radicand = match ctx.get(args[0]) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            sqrt_args[0]
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => *base,
        _ => return None,
    };

    let scale = positive_scaled_variable_factor(ctx, radicand, var_name)
        .or_else(|| nonzero_affine_variable_derivative(ctx, radicand, var_name))?;
    Some((radicand, scale))
}

fn rational_const_for_calculus_presentation(ctx: &mut Context, value: BigRational) -> ExprId {
    if value == BigRational::one() {
        ctx.num(1)
    } else {
        ctx.add(Expr::Number(value))
    }
}

fn nonzero_rational_parts(value: &BigRational) -> Option<(BigRational, BigRational)> {
    if value.is_zero() {
        return None;
    }

    let numerator = BigRational::from_integer(value.numer().clone());
    let denominator = BigRational::from_integer(value.denom().clone());
    Some((numerator, denominator))
}

fn add_one_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    let one = ctx.num(1);
    let raw = ctx.add(Expr::Add(expr, one));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 1,
        max_pow_exp: 1,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(crate) fn try_post_calculus_presentation(
    ctx: &mut Context,
    source: ExprId,
    _result: ExprId,
) -> Option<ExprId> {
    let call = try_extract_diff_call(ctx, source)?;
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    if let Some(compact) = reciprocal_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)
    {
        return Some(compact);
    }

    let (radicand, derivative_scale) =
        arctan_sqrt_scaled_variable_arg(ctx, target, &call.var_name)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let half = BigRational::new(1.into(), 2.into());
    let coefficient = derivative_scale * half;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let denominator_head = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add(Expr::Mul(denominator_scale, sqrt_radicand))
    };
    let radicand_plus_one = add_one_for_calculus_presentation(ctx, radicand);
    let denominator = ctx.add(Expr::Mul(denominator_head, radicand_plus_one));
    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some(compact)
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

fn shifted_sqrt_positive_constant_parts(
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

fn reciprocal_positive_shifted_sqrt_derivative(
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

fn arctan_surd_quotient_scaled_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (arctan_expr, outer_den) = match ctx.get(target).clone() {
        Expr::Div(arctan_expr, outer_den) => (arctan_expr, outer_den),
        _ => return None,
    };
    let outer_radicand = extract_square_root_base(ctx, outer_den)?;
    let outer_radicand_value = cas_ast::views::as_rational_const(ctx, outer_radicand, 8)?;
    if !outer_radicand_value.is_positive() {
        return None;
    }

    let (fn_id, args) = match ctx.get(arctan_expr).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (num, inner_radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    if compare_expr(ctx, outer_radicand, inner_radicand) != std::cmp::Ordering::Equal {
        return None;
    }

    let d_num = differentiate(ctx, num, var_name)?;
    let num_square = squared_expr(ctx, num);
    let denominator = ctx.add(Expr::Add(outer_radicand, num_square));
    Some(ctx.add(Expr::Div(d_num, denominator)))
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

fn fold_numeric_mul_constants_for_hold(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(_, _) => {
            let mut factors = cas_math::expr_nary::mul_leaves(ctx, expr);
            let mut scale = BigRational::one();
            let mut non_numeric = Vec::new();

            while let Some(factor) = factors.pop() {
                let folded = fold_numeric_mul_constants_for_hold(ctx, factor);
                if matches!(ctx.get(folded), Expr::Mul(_, _)) {
                    factors.extend(cas_math::expr_nary::mul_leaves(ctx, folded));
                    continue;
                }
                if let Some(value) = rational_const_for_hold(ctx, folded) {
                    scale *= value;
                } else {
                    non_numeric.push(folded);
                }
            }

            if scale.is_zero() {
                return ctx.num(0);
            }

            if !scale.is_one() && non_numeric.len() == 1 {
                if let Expr::Div(num, den) = ctx.get(non_numeric[0]).clone() {
                    let scale_expr = ctx.add(Expr::Number(scale));
                    let scaled_num = ctx.add(Expr::Mul(scale_expr, num));
                    let folded_num = fold_numeric_mul_constants_for_hold(ctx, scaled_num);
                    return ctx.add(Expr::Div(folded_num, den));
                }
            }

            if !scale.is_one() || non_numeric.is_empty() {
                non_numeric.insert(0, ctx.add(Expr::Number(scale)));
            }

            if non_numeric.len() == 1 {
                non_numeric[0]
            } else {
                cas_math::expr_nary::build_balanced_mul(ctx, &non_numeric)
            }
        }
        Expr::Div(num, den) => {
            let num = fold_numeric_mul_constants_for_hold(ctx, num);
            let den = fold_numeric_mul_constants_for_hold(ctx, den);
            if let Some(den_value) = rational_const_for_hold(ctx, den) {
                if den_value.is_zero() {
                    return ctx.add(Expr::Div(num, den));
                }
                if let Some(num_value) = rational_const_for_hold(ctx, num) {
                    return ctx.add(Expr::Number(num_value / den_value));
                }
                let reciprocal = ctx.add(Expr::Number(BigRational::one() / den_value));
                let scaled = ctx.add(Expr::Mul(reciprocal, num));
                return fold_numeric_mul_constants_for_hold(ctx, scaled);
            }
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner = fold_numeric_mul_constants_for_hold(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn rational_const_for_hold(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Div(num, den) => {
            let num = rational_const_for_hold(ctx, *num)?;
            let den = rational_const_for_hold(ctx, *den)?;
            (!den.is_zero()).then_some(num / den)
        }
        Expr::Neg(inner) => rational_const_for_hold(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

fn inverse_sqrt_quotient_arg_result(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arctan | BuiltinFn::Asinh
                    )
                ) =>
        {
            matches!(ctx.get(args[0]), Expr::Div(_, den) if extract_square_root_base(ctx, *den).is_some())
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            inverse_sqrt_quotient_arg_result(ctx, *left)
                || inverse_sqrt_quotient_arg_result(ctx, *right)
        }
        Expr::Neg(inner) => inverse_sqrt_quotient_arg_result(ctx, *inner),
        _ => false,
    }
}

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    let call = try_extract_integrate_call(ctx, expr)?;
    let required_nonzero = integrate_required_nonzero_conditions(ctx, call.target, &call.var_name);
    let mut required_positive =
        integrate_required_positive_conditions(ctx, call.target, &call.var_name);
    let preserve_compact_reciprocal = cas_math::symbolic_integration_support::integrate_symbolic_is_reciprocal_negative_power_denominator_quotient_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_arctan_reciprocal_affine = cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_reciprocal_affine_variable_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_atanh_polynomial = cas_math::symbolic_integration_support::integrate_symbolic_is_atanh_polynomial_substitution_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_asinh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_affine_variable_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_atanh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_atanh_affine_variable_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_acosh_affine =
        cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_affine_variable_target(
            ctx,
            call.target,
            &call.var_name,
        );
    let preserve_compact_bounded_inverse_trig = cas_math::symbolic_integration_support::integrate_symbolic_is_bounded_inverse_trig_variable_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_trig_polynomial = cas_math::symbolic_integration_support::integrate_symbolic_is_trig_polynomial_substitution_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let preserve_compact_log_cube_product = cas_math::symbolic_integration_support::integrate_symbolic_is_log_cube_product_substitution_target(
        ctx,
        call.target,
        &call.var_name,
    );
    let mut result = integrate(ctx, call.target, &call.var_name)?;
    if required_positive.is_empty() {
        required_positive.extend(collect_atanh_open_interval_conditions(ctx, result));
    }
    let preserve_compact_inverse_sqrt_arg = inverse_sqrt_quotient_arg_result(ctx, result);
    if preserve_compact_reciprocal
        || preserve_compact_arctan_reciprocal_affine
        || preserve_compact_atanh_polynomial
        || preserve_compact_asinh_affine
        || preserve_compact_atanh_affine
        || preserve_compact_acosh_affine
        || preserve_compact_bounded_inverse_trig
        || preserve_compact_trig_polynomial
        || preserve_compact_log_cube_product
        || preserve_compact_inverse_sqrt_arg
    {
        result = fold_numeric_mul_constants_for_hold(ctx, result);
        result = cas_ast::hold::wrap_hold(ctx, result);
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
    let target = unwrap_internal_hold_for_calculus(ctx, call.target);
    let result = reciprocal_positive_shifted_sqrt_derivative(ctx, target, &call.var_name)
        .or_else(|| {
            bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, &call.var_name)
        })
        .or_else(|| asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| arctan_surd_quotient_scaled_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| atanh_surd_quotient_compact_derivative(ctx, target, &call.var_name))
        .or_else(|| differentiate(ctx, target, &call.var_name))?;
    let required_conditions = atanh_diff_required_conditions(ctx, target);
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

#[cfg(test)]
mod compact_hold_tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_collapses_rational_noise() {
        let mut ctx = Context::new();
        let expr = parse("(atanh(x^2/sqrt(3)) * 1/2 * 2)/sqrt(3)", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "atanh(x^2 / sqrt(3)) / sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_absorbs_outer_scale_into_quotient() {
        let mut ctx = Context::new();
        let expr = parse("2 * ((atanh(x^2/sqrt(3))/2)/sqrt(3))", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "atanh(x^2 / sqrt(3)) / sqrt(3)");
    }

    #[test]
    fn inverse_sqrt_quotient_arg_result_detects_compact_inverse_sqrt_substitution() {
        let mut ctx = Context::new();
        let expr = parse("arcsin(x^2/sqrt(3))", &mut ctx).unwrap();

        assert!(inverse_sqrt_quotient_arg_result(&ctx, expr));

        let rationalized = parse("arcsin(1/3 * sqrt(3) * x^2)", &mut ctx).unwrap();

        assert!(!inverse_sqrt_quotient_arg_result(&ctx, rationalized));

        let arctan = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut ctx).unwrap();

        assert!(inverse_sqrt_quotient_arg_result(&ctx, arctan));
    }

    #[test]
    fn arctan_surd_quotient_scaled_compact_derivative_avoids_rationalized_route() {
        let mut ctx = Context::new();
        let expr = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut ctx).unwrap();
        let derivative =
            arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / ((2 * x + 2)^2 + 6)");
    }
}

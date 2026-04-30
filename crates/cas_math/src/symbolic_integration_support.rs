//! Symbolic integration helpers shared by integration-facing rule layers.

use crate::build::mul2_raw;
use crate::expr_nary::{build_balanced_mul, mul_leaves};
use crate::expr_predicates::contains_named_var;
use crate::polynomial::Polynomial;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;

fn ln_abs(ctx: &mut Context, arg: ExprId) -> ExprId {
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    ctx.call_builtin(BuiltinFn::Ln, vec![abs_arg])
}

fn is_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(value.into()))
}

fn is_negative_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == BigRational::new((-1).into(), 2.into()),
        Expr::Div(num, den) => is_number(ctx, *num, -1) && is_number(ctx, *den, 2),
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Number(n) => *n == BigRational::new(1.into(), 2.into()),
            Expr::Div(num, den) => is_number(ctx, *num, 1) && is_number(ctx, *den, 2),
            _ => false,
        },
        _ => false,
    }
}

fn is_positive_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => is_number(ctx, *num, 1) && is_number(ctx, *den, 2),
        _ => false,
    }
}

fn sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) if is_positive_half(ctx, *exp) => Some(*base),
        _ => None,
    }
}

fn reciprocal_sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_negative_half(ctx, *exp) => Some(*base),
        _ => None,
    }
}

fn is_var_square(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => is_var(ctx, *base, var) && is_number(ctx, *exp, 2),
        _ => false,
    }
}

fn is_var_square_plus_one(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            (is_var_square(ctx, *l, var) && is_number(ctx, *r, 1))
                || (is_number(ctx, *l, 1) && is_var_square(ctx, *r, var))
        }
        _ => false,
    }
}

fn reciprocal_trig_square_parts(ctx: &Context, den: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let (base, exp) = match ctx.get(den) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    if !is_number(ctx, exp, 2) {
        return None;
    }

    let (builtin, arg) = match ctx.get(base) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    match builtin {
        BuiltinFn::Cos | BuiltinFn::Sin => Some((builtin, arg)),
        _ => None,
    }
}

fn reciprocal_trig_square_antiderivative(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;

    let integral = match builtin {
        BuiltinFn::Cos => {
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            ctx.add(Expr::Div(sin_arg, cos_arg))
        }
        BuiltinFn::Sin => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cot_arg = ctx.add(Expr::Div(cos_arg, sin_arg));
            ctx.add(Expr::Neg(cot_arg))
        }
        _ => return None,
    };

    let is_a_one = if let Expr::Number(n) = ctx.get(a) {
        n.is_one()
    } else {
        false
    };
    if is_a_one {
        Some(integral)
    } else {
        Some(ctx.add(Expr::Div(integral, a)))
    }
}

fn polynomial_reciprocal_trig_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let integral = match builtin {
        BuiltinFn::Cos => {
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            ctx.add(Expr::Div(sin_arg, cos_arg))
        }
        BuiltinFn::Sin => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cot_arg = ctx.add(Expr::Div(cos_arg, sin_arg));
            ctx.add(Expr::Neg(cot_arg))
        }
        _ => return None,
    };

    if scale.is_one() {
        return Some(integral);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

fn reciprocal_hyperbolic_square_arg(
    ctx: &Context,
    den: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let (base, exp) = match ctx.get(den) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    if !is_number(ctx, exp, 2) {
        return None;
    }

    match ctx.get(base) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn reciprocal_hyperbolic_cosh_square_arg(ctx: &Context, den: ExprId) -> Option<ExprId> {
    reciprocal_hyperbolic_square_arg(ctx, den, BuiltinFn::Cosh)
}

fn reciprocal_hyperbolic_sinh_square_arg(ctx: &Context, den: ExprId) -> Option<ExprId> {
    reciprocal_hyperbolic_square_arg(ctx, den, BuiltinFn::Sinh)
}

fn hyperbolic_tanh_reciprocal_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = reciprocal_hyperbolic_cosh_square_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
    if scale.is_one() {
        return Some(tanh_arg);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, tanh_arg))
}

fn hyperbolic_coth_reciprocal_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = reciprocal_hyperbolic_sinh_square_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let quotient = ctx.add(Expr::Div(cosh_arg, sinh_arg));
    if scale.is_one() {
        return Some(ctx.add(Expr::Neg(quotient)));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    let negative_scale_expr = ctx.add(Expr::Neg(scale_expr));
    Some(mul2_raw(ctx, negative_scale_expr, quotient))
}

fn hyperbolic_log_derivative_ratio_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = match ctx.get(den) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cosh => BuiltinFn::Sinh,
        BuiltinFn::Sinh => BuiltinFn::Cosh,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let log_abs_den = ln_abs(ctx, den);
    if scale.is_one() {
        return Some(log_abs_den);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, log_abs_den))
}

fn trig_log_derivative_ratio_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = match ctx.get(den) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let scaled = match den_builtin {
        BuiltinFn::Cos => -scale,
        BuiltinFn::Sin => scale,
        _ => return None,
    };
    let log_abs_den = ln_abs(ctx, den);
    if scaled.is_one() {
        return Some(log_abs_den);
    }
    if scaled == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(log_abs_den)));
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, log_abs_den))
}

fn hyperbolic_tanh_reciprocal_log_sinh_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = unary_builtin_arg(ctx, den, BuiltinFn::Tanh)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let log_abs_sinh = ln_abs(ctx, sinh_arg);
    if scale.is_one() {
        return Some(log_abs_sinh);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, log_abs_sinh))
}

fn hyperbolic_cosh_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = reciprocal_hyperbolic_cosh_square_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (sinh_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, BuiltinFn::Sinh)
            .is_some_and(|sinh_arg| compare_expr(ctx, sinh_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sinh_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let minus_one = BigRational::from_integer((-1).into());
    let minus_one_expr = ctx.add(Expr::Number(minus_one));
    let reciprocal_cosh = ctx.add(Expr::Pow(cosh_arg, minus_one_expr));
    let scale_expr = ctx.add(Expr::Number(scale));
    let negative_scale_expr = ctx.add(Expr::Neg(scale_expr));
    Some(mul2_raw(ctx, negative_scale_expr, reciprocal_cosh))
}

fn hyperbolic_sinh_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg = reciprocal_hyperbolic_sinh_square_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (cosh_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, BuiltinFn::Cosh)
            .is_some_and(|cosh_arg| compare_expr(ctx, cosh_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != cosh_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let minus_one = BigRational::from_integer((-1).into());
    let minus_one_expr = ctx.add(Expr::Number(minus_one));
    let reciprocal_sinh = ctx.add(Expr::Pow(sinh_arg, minus_one_expr));
    let scale_expr = ctx.add(Expr::Number(scale));
    let negative_scale_expr = ctx.add(Expr::Neg(scale_expr));
    Some(mul2_raw(ctx, negative_scale_expr, reciprocal_sinh))
}

fn hyperbolic_reciprocal_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    hyperbolic_tanh_reciprocal_square_antiderivative(ctx, num, den, var)
        .or_else(|| hyperbolic_coth_reciprocal_square_antiderivative(ctx, num, den, var))
        .or_else(|| hyperbolic_cosh_reciprocal_derivative_antiderivative(ctx, num, den, var))
        .or_else(|| hyperbolic_sinh_reciprocal_derivative_antiderivative(ctx, num, den, var))
}

fn constant_scaled_hyperbolic_reciprocal_square_antiderivative(
    ctx: &mut Context,
    constant: ExprId,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let scaled_num = mul2_raw(ctx, constant, num);
    hyperbolic_reciprocal_square_antiderivative(ctx, scaled_num, den, var)
}

fn reciprocal_trig_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    if !is_number(ctx, num, 1) {
        return None;
    }

    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    get_linear_coeffs(ctx, arg, var)?;
    Some(match builtin {
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Cos, vec![arg]),
        BuiltinFn::Sin => ctx.call_builtin(BuiltinFn::Sin, vec![arg]),
        _ => return None,
    })
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn divide_by_coeff_unless_one(ctx: &mut Context, integral: ExprId, coeff: ExprId) -> ExprId {
    let is_coeff_one = if let Expr::Number(n) = ctx.get(coeff) {
        n.is_one()
    } else {
        false
    };

    if is_coeff_one {
        integral
    } else {
        ctx.add(Expr::Div(integral, coeff))
    }
}

fn trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    let numerator_arg = unary_builtin_arg(ctx, num, numerator_builtin)?;
    if compare_expr(ctx, numerator_arg, arg) != Ordering::Equal {
        return None;
    }

    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let integral = match den_builtin {
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Sec, vec![arg]),
        BuiltinFn::Sin => {
            let csc_arg = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
            ctx.add(Expr::Neg(csc_arg))
        }
        _ => return None,
    };
    Some(divide_by_coeff_unless_one(ctx, integral, a))
}

fn polynomial_trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let integral = match den_builtin {
        BuiltinFn::Cos => {
            let one = ctx.num(1);
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            ctx.add(Expr::Div(one, cos_arg))
        }
        BuiltinFn::Sin => {
            let one = ctx.num(1);
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let reciprocal_sin = ctx.add(Expr::Div(one, sin_arg));
            ctx.add(Expr::Neg(reciprocal_sin))
        }
        _ => return None,
    };
    if scale.is_one() {
        return Some(integral);
    }
    if scale == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(integral)));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

fn constant_scaled_trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    constant: ExprId,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let scaled_num = mul2_raw(ctx, constant, num);
    polynomial_trig_reciprocal_derivative_antiderivative(ctx, scaled_num, den, var)
}

fn trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    let numerator_arg = unary_builtin_arg(ctx, num, numerator_builtin)?;
    if compare_expr(ctx, numerator_arg, arg) != Ordering::Equal {
        return None;
    }
    get_linear_coeffs(ctx, arg, var)?;
    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let factors = mul_leaves(ctx, num);
    let (numerator_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, numerator_builtin)
            .is_some_and(|numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal)
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
        .collect();
    if cofactor_factors.is_empty() {
        return None;
    }
    let cofactor = build_balanced_mul(ctx, &cofactor_factors);

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    Some(ctx.call_builtin(den_builtin, vec![arg]))
}

fn polynomial_trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(ctx, num, den, var)
}

fn constant_scaled_trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (constant, expr) = match ctx.get(expr).clone() {
        Expr::Mul(l, r) if !contains_named_var(ctx, l, var) => (l, r),
        Expr::Mul(l, r) if !contains_named_var(ctx, r, var) => (r, l),
        _ => return None,
    };
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let scaled_num = mul2_raw(ctx, constant, num);
    polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(ctx, scaled_num, den, var)
}

fn trig_log_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let log_arg_builtin = match builtin {
        BuiltinFn::Tan => BuiltinFn::Cos,
        BuiltinFn::Cot => BuiltinFn::Sin,
        _ => return None,
    };
    let log_arg = ctx.call_builtin(log_arg_builtin, vec![arg]);
    let log_abs = ln_abs(ctx, log_arg);
    let integral = match builtin {
        BuiltinFn::Tan => ctx.add(Expr::Neg(log_abs)),
        BuiltinFn::Cot => log_abs,
        _ => return None,
    };

    let is_a_one = if let Expr::Number(n) = ctx.get(a) {
        n.is_one()
    } else {
        false
    };
    if is_a_one {
        Some(integral)
    } else {
        Some(ctx.add(Expr::Div(integral, a)))
    }
}

fn trig_log_required_nonzero(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let (builtin, arg) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(fn_id)?, args[0]),
        _ => return None,
    };
    let nonzero_builtin = match builtin {
        BuiltinFn::Tan => BuiltinFn::Cos,
        BuiltinFn::Cot => BuiltinFn::Sin,
        _ => return None,
    };
    get_linear_coeffs(ctx, arg, var)?;
    Some(ctx.call_builtin(nonzero_builtin, vec![arg]))
}

#[derive(Clone, Copy)]
enum PolynomialSubstitutionKernel {
    Exp,
    Sin,
    Cos,
    Sinh,
    Cosh,
    Tanh,
    Tan,
    Cot,
}

fn polynomial_substitution_kernel(
    ctx: &Context,
    expr: ExprId,
) -> Option<(PolynomialSubstitutionKernel, ExprId)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let kernel = match ctx.builtin_of(*fn_id)? {
                BuiltinFn::Exp => PolynomialSubstitutionKernel::Exp,
                BuiltinFn::Sin => PolynomialSubstitutionKernel::Sin,
                BuiltinFn::Cos => PolynomialSubstitutionKernel::Cos,
                BuiltinFn::Sinh => PolynomialSubstitutionKernel::Sinh,
                BuiltinFn::Cosh => PolynomialSubstitutionKernel::Cosh,
                BuiltinFn::Tanh => PolynomialSubstitutionKernel::Tanh,
                BuiltinFn::Tan => PolynomialSubstitutionKernel::Tan,
                BuiltinFn::Cot => PolynomialSubstitutionKernel::Cot,
                _ => return None,
            };
            Some((kernel, args[0]))
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => {
            Some((PolynomialSubstitutionKernel::Exp, *exp))
        }
        _ => None,
    }
}

fn polynomial_substitution_kernel_antiderivative(
    ctx: &mut Context,
    kernel: PolynomialSubstitutionKernel,
    arg: ExprId,
    kernel_factor: ExprId,
) -> ExprId {
    match kernel {
        PolynomialSubstitutionKernel::Exp => kernel_factor,
        PolynomialSubstitutionKernel::Sin => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            ctx.add(Expr::Neg(cos_arg))
        }
        PolynomialSubstitutionKernel::Cos => ctx.call_builtin(BuiltinFn::Sin, vec![arg]),
        PolynomialSubstitutionKernel::Sinh => ctx.call_builtin(BuiltinFn::Cosh, vec![arg]),
        PolynomialSubstitutionKernel::Cosh => ctx.call_builtin(BuiltinFn::Sinh, vec![arg]),
        PolynomialSubstitutionKernel::Tanh => {
            let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
            ln_abs(ctx, cosh_arg)
        }
        PolynomialSubstitutionKernel::Tan => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            let log_abs = ln_abs(ctx, cos_arg);
            ctx.add(Expr::Neg(log_abs))
        }
        PolynomialSubstitutionKernel::Cot => {
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            ln_abs(ctx, sin_arg)
        }
    }
}

fn constant_polynomial_ratio(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<BigRational> {
    if denominator.is_zero() {
        return None;
    }

    let pivot = denominator
        .coeffs
        .iter()
        .position(|coeff| !coeff.is_zero())?;
    let numerator_pivot = numerator
        .coeffs
        .get(pivot)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let scale = numerator_pivot / denominator.coeffs[pivot].clone();
    let len = numerator.coeffs.len().max(denominator.coeffs.len());

    for idx in 0..len {
        let left = numerator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = denominator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            * scale.clone();
        if left != right {
            return None;
        }
    }

    Some(scale)
}

fn exact_rational_sqrt(value: &BigRational) -> Option<BigRational> {
    if value < &BigRational::zero() {
        return None;
    }

    let sqrt_num = value.numer().sqrt();
    let sqrt_den = value.denom().sqrt();
    if &sqrt_num * &sqrt_num == value.numer().clone()
        && &sqrt_den * &sqrt_den == value.denom().clone()
    {
        Some(BigRational::new(sqrt_num, sqrt_den))
    } else {
        None
    }
}

fn exact_polynomial_square_plus_positive_constant(
    poly: &Polynomial,
) -> Option<(Polynomial, BigRational)> {
    if poly.is_zero() {
        return None;
    }

    let degree = poly.degree();
    if degree == 0 || !degree.is_multiple_of(2) {
        return None;
    }

    let root_degree = degree / 2;
    let leading = poly
        .coeffs
        .get(degree)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let leading_root = exact_rational_sqrt(&leading)?;
    if leading_root.is_zero() {
        return None;
    }

    let mut root_coeffs = vec![BigRational::zero(); root_degree + 1];
    root_coeffs[root_degree] = leading_root.clone();
    let two = BigRational::from_integer(2.into());

    for k in (0..root_degree).rev() {
        let target_degree = root_degree + k;
        let target = poly
            .coeffs
            .get(target_degree)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let mut known = BigRational::zero();

        for i in 0..=root_degree {
            if let Some(j) = target_degree.checked_sub(i) {
                if j <= root_degree && i != k && j != k {
                    known += root_coeffs[i].clone() * root_coeffs[j].clone();
                }
            }
        }

        root_coeffs[k] = (target - known) / (two.clone() * leading_root.clone());
    }

    let root = Polynomial::new(root_coeffs, poly.var.clone());
    let square = root.mul(&root);
    let len = poly.coeffs.len().max(square.coeffs.len());

    for idx in 1..len {
        let left = poly
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = square
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if left != right {
            return None;
        }
    }

    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero)
        - square
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
    if constant > BigRational::zero() {
        Some((root, constant))
    } else {
        None
    }
}

fn exact_positive_constant_minus_polynomial_square(
    poly: &Polynomial,
) -> Option<(Polynomial, BigRational)> {
    if poly.is_zero() {
        return None;
    }

    let degree = poly.degree();
    if degree == 0 || !degree.is_multiple_of(2) {
        return None;
    }

    let root_degree = degree / 2;
    let leading = poly
        .coeffs
        .get(degree)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let leading_root = exact_rational_sqrt(&(-leading))?;
    if leading_root.is_zero() {
        return None;
    }

    let mut root_coeffs = vec![BigRational::zero(); root_degree + 1];
    root_coeffs[root_degree] = leading_root.clone();
    let two = BigRational::from_integer(2.into());

    for k in (0..root_degree).rev() {
        let target_degree = root_degree + k;
        let target = -poly
            .coeffs
            .get(target_degree)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let mut known = BigRational::zero();

        for i in 0..=root_degree {
            if let Some(j) = target_degree.checked_sub(i) {
                if j <= root_degree && i != k && j != k {
                    known += root_coeffs[i].clone() * root_coeffs[j].clone();
                }
            }
        }

        root_coeffs[k] = (target - known) / (two.clone() * leading_root.clone());
    }

    let root = Polynomial::new(root_coeffs, poly.var.clone());
    let square = root.mul(&root);
    let len = poly.coeffs.len().max(square.coeffs.len());

    for idx in 1..len {
        let left = poly
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = square
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if left != -right {
            return None;
        }
    }

    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero)
        + square
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
    if constant > BigRational::zero() {
        Some((root, constant))
    } else {
        None
    }
}

fn polynomial_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (kernel_index, kernel, kernel_arg) =
        factors.iter().enumerate().find_map(|(idx, factor)| {
            polynomial_substitution_kernel(ctx, *factor).map(|(kernel, arg)| (idx, kernel, arg))
        })?;

    if !contains_named_var(ctx, kernel_arg, var) {
        return None;
    }

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != kernel_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        if matches!(
            kernel,
            PolynomialSubstitutionKernel::Tan | PolynomialSubstitutionKernel::Cot
        ) {
            return None;
        }
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, kernel_arg, var).ok()?;
    let derivative_poly = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative_poly)?;
    if scale.is_zero() {
        return None;
    }

    let antiderivative = polynomial_substitution_kernel_antiderivative(
        ctx,
        kernel,
        kernel_arg,
        factors[kernel_index],
    );
    if scale.is_one() {
        return Some(antiderivative);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, antiderivative))
}

fn arctan_polynomial_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) = exact_polynomial_square_plus_positive_constant(&denominator)?;
    let offset = exact_rational_sqrt(&offset_square)?;
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let arctan_arg = if offset.is_one() {
        arg
    } else {
        let offset_expr = ctx.add(Expr::Number(offset.clone()));
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    let scaled = scale / offset;
    if scaled.is_one() {
        return Some(arctan);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, arctan))
}

fn atanh_polynomial_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) = exact_positive_constant_minus_polynomial_square(&denominator)?;
    let offset = exact_rational_sqrt(&offset_square)?;
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let atanh_arg = if offset.is_one() {
        arg
    } else {
        let offset_expr = ctx.add(Expr::Number(offset.clone()));
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let atanh = ctx.call_builtin(BuiltinFn::Atanh, vec![atanh_arg]);
    let scaled = scale / offset;
    if scaled.is_one() {
        return Some(atanh);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, atanh))
}

fn polynomial_square_minus_constant_log_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) =
        exact_positive_constant_minus_polynomial_square(&denominator.neg())?;
    let offset = exact_rational_sqrt(&offset_square)?;
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let offset_expr = ctx.add(Expr::Number(offset.clone()));
    let numerator_arg = ctx.add(Expr::Sub(arg, offset_expr));
    let denominator_arg = ctx.add(Expr::Add(arg, offset_expr));
    let ratio = ctx.add(Expr::Div(numerator_arg, denominator_arg));
    let log_abs_ratio = ln_abs(ctx, ratio);

    let two = BigRational::from_integer(2.into());
    let scaled = scale / (two * offset);
    if scaled.is_one() {
        return Some(log_abs_ratio);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, log_abs_ratio))
}

fn polynomial_log_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    if denominator.degree() == 0 {
        return None;
    }

    let derivative = denominator.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let log_abs_den = ln_abs(ctx, den);
    if scale.is_one() {
        return Some(log_abs_den);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, log_abs_den))
}

fn arcsin_polynomial_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let (arg_poly, offset_square) = exact_positive_constant_minus_polynomial_square(&radicand)?;
    let offset = exact_rational_sqrt(&offset_square)?;
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let arcsin_arg = if offset.is_one() {
        arg
    } else {
        let offset_expr = ctx.add(Expr::Number(offset));
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let arcsin = ctx.call_builtin(BuiltinFn::Arcsin, vec![arcsin_arg]);
    if scale.is_one() {
        return Some(arcsin);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, arcsin))
}

fn asinh_polynomial_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let (arg_poly, offset_square) = exact_polynomial_square_plus_positive_constant(&radicand)?;
    let offset = exact_rational_sqrt(&offset_square)?;
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let asinh_arg = if offset.is_one() {
        arg
    } else {
        let offset_expr = ctx.add(Expr::Number(offset));
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let asinh = ctx.call_builtin(BuiltinFn::Asinh, vec![asinh_arg]);
    if scale.is_one() {
        return Some(asinh);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, asinh))
}

fn arcsin_polynomial_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        return arcsin_polynomial_substitution_from_radicand(ctx, num, radicand, var);
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn asinh_polynomial_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        return asinh_polynomial_substitution_from_radicand(ctx, num, radicand, var);
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    asinh_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn arcsin_polynomial_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn asinh_polynomial_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    asinh_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

fn arcsin_polynomial_substitution_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            if let Some(radicand) = sqrt_like_radicand(ctx, den) {
                arcsin_polynomial_substitution_from_radicand(ctx, num, radicand, var)?;
                return Some(radicand);
            }

            let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
            let factors = mul_leaves(ctx, num);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                let radicand = sqrt_like_radicand(ctx, *factor)?;
                let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
                (radicand_poly == denominator).then_some((idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        _ => None,
    }
}

fn atanh_polynomial_substitution_denominator(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    atanh_polynomial_substitution_antiderivative(ctx, num, den, var)?;
    Some(den)
}

pub fn integrate_symbolic_required_nonzero_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    trig_log_required_nonzero(ctx, expr, var)
        .into_iter()
        .chain(trig_reciprocal_derivative_required_nonzero(ctx, expr, var))
        .chain(polynomial_trig_reciprocal_derivative_required_nonzero(
            ctx, expr, var,
        ))
        .chain(constant_scaled_trig_reciprocal_derivative_required_nonzero(
            ctx, expr, var,
        ))
        .chain(reciprocal_trig_square_required_nonzero(ctx, expr, var))
        .collect()
}

pub fn integrate_symbolic_required_positive_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let mut conditions: Vec<ExprId> = arcsin_polynomial_substitution_radicand(ctx, expr, var)
        .into_iter()
        .collect();
    conditions.extend(atanh_polynomial_substitution_denominator(ctx, expr, var));
    conditions
}

/// Integrate `expr` with respect to `var` using a small set of symbolic rules.
pub fn integrate_symbolic_expr(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    // Extract variant info in one borrow, then process with owned ExprId values.
    enum IntKind {
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Neg(ExprId),
        Mul(ExprId, ExprId),
        Pow(ExprId, ExprId),
        Variable(usize),
        Div(ExprId, ExprId),
        Function(usize, Vec<ExprId>),
        Other,
    }
    let kind = match ctx.get(expr) {
        Expr::Add(l, r) => IntKind::Add(*l, *r),
        Expr::Sub(l, r) => IntKind::Sub(*l, *r),
        Expr::Neg(inner) => IntKind::Neg(*inner),
        Expr::Mul(l, r) => IntKind::Mul(*l, *r),
        Expr::Pow(b, e) => IntKind::Pow(*b, *e),
        Expr::Variable(s) => IntKind::Variable(*s),
        Expr::Div(n, d) => IntKind::Div(*n, *d),
        Expr::Function(f, args) => IntKind::Function(*f, args.clone()),
        _ => IntKind::Other,
    };

    if let IntKind::Add(l, r) = kind {
        let int_l = integrate_symbolic_expr(ctx, l, var)?;
        let int_r = integrate_symbolic_expr(ctx, r, var)?;
        return Some(ctx.add(Expr::Add(int_l, int_r)));
    }

    if let IntKind::Sub(l, r) = kind {
        let int_l = integrate_symbolic_expr(ctx, l, var)?;
        let int_r = integrate_symbolic_expr(ctx, r, var)?;
        return Some(ctx.add(Expr::Sub(int_l, int_r)));
    }

    if let IntKind::Neg(inner) = kind {
        let inner_integral = integrate_symbolic_expr(ctx, inner, var)?;
        return Some(ctx.add(Expr::Neg(inner_integral)));
    }

    if let Some(integral) = polynomial_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = arcsin_polynomial_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = asinh_polynomial_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let IntKind::Mul(l, r) = kind {
        if !contains_named_var(ctx, l, var) {
            if let Some(integral) =
                constant_scaled_trig_reciprocal_derivative_antiderivative(ctx, l, r, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_hyperbolic_reciprocal_square_antiderivative(ctx, l, r, var)
            {
                return Some(integral);
            }
            if let Some(int_r) = integrate_symbolic_expr(ctx, r, var) {
                return Some(mul2_raw(ctx, l, int_r));
            }
        }
        if !contains_named_var(ctx, r, var) {
            if let Some(integral) =
                constant_scaled_trig_reciprocal_derivative_antiderivative(ctx, r, l, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_hyperbolic_reciprocal_square_antiderivative(ctx, r, l, var)
            {
                return Some(integral);
            }
            if let Some(int_l) = integrate_symbolic_expr(ctx, l, var) {
                return Some(mul2_raw(ctx, r, int_l));
            }
        }
    }

    if !contains_named_var(ctx, expr, var) {
        let var_expr = ctx.var(var);
        return Some(mul2_raw(ctx, expr, var_expr));
    }

    if let IntKind::Pow(base, exp) = kind {
        if is_negative_half(ctx, exp) && is_var_square_plus_one(ctx, base, var) {
            let var_expr = ctx.var(var);
            return Some(ctx.call_builtin(BuiltinFn::Asinh, vec![var_expr]));
        }

        if let Some((a, _)) = get_linear_coeffs(ctx, base, var) {
            if !contains_named_var(ctx, exp, var) {
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == BigRational::from_integer((-1).into()) {
                        let ln_u = ln_abs(ctx, base);
                        return Some(ctx.add(Expr::Div(ln_u, a)));
                    }
                }

                let one = ctx.num(1);
                let new_exp = ctx.add(Expr::Add(exp, one));

                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };
                let new_denom = if is_a_one {
                    new_exp
                } else {
                    mul2_raw(ctx, a, new_exp)
                };

                let pow_expr = ctx.add(Expr::Pow(base, new_exp));
                return Some(ctx.add(Expr::Div(pow_expr, new_denom)));
            }
        }

        if !contains_named_var(ctx, base, var) {
            if let Some((a, _)) = get_linear_coeffs(ctx, exp, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                let is_e = if let Expr::Constant(c) = ctx.get(base) {
                    c == &cas_ast::Constant::E
                } else {
                    false
                };

                if is_e {
                    if is_a_one {
                        return Some(expr);
                    }
                    return Some(ctx.add(Expr::Div(expr, a)));
                }

                let ln_c = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let denom = if is_a_one {
                    ln_c
                } else {
                    mul2_raw(ctx, a, ln_c)
                };
                return Some(ctx.add(Expr::Div(expr, denom)));
            }
        }
    }

    if let IntKind::Variable(sym_id) = kind {
        if ctx.sym_name(sym_id) == var {
            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let pow_expr = ctx.add(Expr::Pow(var_expr, two));
            return Some(ctx.add(Expr::Div(pow_expr, two)));
        }
    }

    if let IntKind::Div(num, den) = kind {
        if let Some(integral) = trig_reciprocal_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_trig_reciprocal_derivative_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = hyperbolic_log_derivative_ratio_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_log_derivative_ratio_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            hyperbolic_tanh_reciprocal_log_sinh_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = hyperbolic_reciprocal_square_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            arcsin_polynomial_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = asinh_polynomial_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = atanh_polynomial_substitution_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_square_minus_constant_log_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = arctan_polynomial_substitution_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Expr::Number(n) = ctx.get(num) {
            if n.is_one() {
                if is_var_square_plus_one(ctx, den, var) {
                    let var_expr = ctx.var(var);
                    return Some(ctx.call_builtin(BuiltinFn::Arctan, vec![var_expr]));
                }

                if let Some(integral) = reciprocal_trig_square_antiderivative(ctx, den, var) {
                    return Some(integral);
                }

                if let Some((a, _)) = get_linear_coeffs(ctx, den, var) {
                    let ln_den = ln_abs(ctx, den);
                    return Some(ctx.add(Expr::Div(ln_den, a)));
                }
            }
        }

        if let Some(integral) = polynomial_reciprocal_trig_square_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = polynomial_log_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }
    }

    if let IntKind::Function(fn_id, args) = kind {
        if args.len() == 1 {
            let arg = args[0];
            if let Some(builtin @ (BuiltinFn::Tan | BuiltinFn::Cot)) = ctx.builtin_of(fn_id) {
                return trig_log_antiderivative(ctx, builtin, arg, var);
            }

            if let Some((a, _)) = get_linear_coeffs(ctx, arg, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                match ctx.builtin_of(fn_id) {
                    Some(BuiltinFn::Sin) => {
                        let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                        let integral = ctx.add(Expr::Neg(cos_arg));
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    Some(BuiltinFn::Cos) => {
                        let integral = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    Some(BuiltinFn::Exp) => {
                        if is_a_one {
                            return Some(expr);
                        }
                        return Some(ctx.add(Expr::Div(expr, a)));
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

/// Returns `(a, b)` such that `expr = a*var + b`.
pub fn get_linear_coeffs(ctx: &mut Context, expr: ExprId, var: &str) -> Option<(ExprId, ExprId)> {
    if !contains_named_var(ctx, expr, var) {
        return Some((ctx.num(0), expr));
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some((ctx.num(1), ctx.num(0))),
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if !contains_named_var(ctx, l, var) && is_var(ctx, r, var) {
                return Some((l, ctx.num(0)));
            }
            if is_var(ctx, l, var) && !contains_named_var(ctx, r, var) {
                return Some((r, ctx.num(0)));
            }
            None
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);

            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_named_var(ctx, a1, var) && !contains_named_var(ctx, a2, var) {
                    let a = ctx.add(Expr::Add(a1, a2));
                    let b = ctx.add(Expr::Add(b1, b2));
                    return Some((a, b));
                }
            }
            None
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);
            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_named_var(ctx, a1, var) && !contains_named_var(ctx, a2, var) {
                    let a = ctx.add(Expr::Sub(a1, a2));
                    let b = ctx.add(Expr::Sub(b1, b2));
                    return Some((a, b));
                }
            }
            None
        }
        _ => None,
    }
}

fn is_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        ctx.sym_name(*sym_id) == var
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{get_linear_coeffs, integrate_symbolic_expr};
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn integrates_simple_power() {
        let mut ctx = Context::new();
        let expr = parse("x^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x^(1 + 2) / (1 + 2)");
    }

    #[test]
    fn integrates_linear_trig_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2 * cos(2 * x)");
    }

    #[test]
    fn integrates_explicit_negation_by_linearity() {
        let mut ctx = Context::new();
        let expr = parse("-sin(x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "cos(x)");

        let expr = parse("-(x*sin(x^2)/cos(x^2)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-((1 * 1/2)/cos(x^2))");

        let expr = parse("-(x^2*cos(x^3)/sin(x^3)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-(-(1 * 1/3)/sin(x^3))");

        let expr = parse("-tan(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
    }

    #[test]
    fn integrates_polynomial_derivative_exp_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*exp(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "e^(x^2)");
    }

    #[test]
    fn integrates_polynomial_derivative_trig_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*cos(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sin(x^2)");

        let expr = parse("2*x*sin(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-cos(x^2)");
    }

    #[test]
    fn integrates_polynomial_derivative_hyperbolic_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sinh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * cosh(2 * x + 1)");

        let expr = parse("cosh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * sinh(2 * x + 1)");

        let expr = parse("2*x*sinh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "cosh(x^2)");

        let expr = parse("2*x*cosh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sinh(x^2)");

        let expr = parse("tanh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|cosh(2 * x + 1)|)");

        let expr = parse("2*x*tanh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|cosh(x^2)|)");
    }

    #[test]
    fn integrates_hyperbolic_log_derivative_ratio_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sinh(2*x + 1)/cosh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|cosh(2 * x + 1)|)");

        let expr = parse("cosh(2*x + 1)/sinh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(2 * x + 1)|)");

        let expr = parse("2*x*cosh(x^2)/sinh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sinh(x^2)|)");
    }

    #[test]
    fn integrates_hyperbolic_tanh_reciprocal_log_sinh_substitution() {
        let mut ctx = Context::new();
        let expr = parse("1/tanh(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(2 * x + 1)|)");

        let expr = parse("2*x/tanh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sinh(x^2)|)");

        let expr = parse("x/tanh(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(x^2)|)");
    }

    #[test]
    fn integrates_hyperbolic_tanh_reciprocal_square_substitution() {
        let mut ctx = Context::new();
        let expr = parse("1/cosh(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * tanh(2 * x + 1)");

        let expr = parse("2*x/cosh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "tanh(x^2)");

        let expr = parse("x/cosh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * tanh(x^2)");
    }

    #[test]
    fn integrates_hyperbolic_coth_reciprocal_square_substitution() {
        let mut ctx = Context::new();
        let expr = parse("1/sinh(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(
            rendered(&ctx, out),
            "(cosh(2 * x + 1) * -1/2)/sinh(2 * x + 1)"
        );

        let expr = parse("2*x/sinh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-(cosh(x^2) / sinh(x^2))");

        let expr = parse("x/sinh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "(cosh(x^2) * -1/2)/sinh(x^2)");
    }

    #[test]
    fn integrates_hyperbolic_cosh_reciprocal_derivative_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sinh(2*x + 1)/cosh(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2/cosh(2 * x + 1)");

        let expr = parse("2*x*sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/cosh(x^2)");

        let expr = parse("x*sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2/cosh(x^2)");
    }

    #[test]
    fn integrates_hyperbolic_sinh_reciprocal_derivative_substitution() {
        let mut ctx = Context::new();
        let expr = parse("cosh(2*x + 1)/sinh(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2/sinh(2 * x + 1)");

        let expr = parse("2*x*cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/sinh(x^2)");

        let expr = parse("x*cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2/sinh(x^2)");
    }

    #[test]
    fn hyperbolic_substitution_rejects_missing_polynomial_cofactor() {
        let mut ctx = Context::new();
        let expr = parse("sinh(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty()
        );
        assert!(
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x").is_empty()
        );

        let expr = parse("tanh(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty()
        );
        assert!(
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x").is_empty()
        );

        let expr = parse("1/cosh(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("cosh(x^2)/sinh(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("1/sinh(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("1/tanh(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

        let expr = parse("1/(x^4-1)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
    }

    #[test]
    fn integrates_reciprocal_linear_with_absolute_log() {
        let mut ctx = Context::new();
        let expr = parse("1/(3*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|3 * x|) / 3");
    }

    #[test]
    fn integrates_linear_power_minus_one_with_absolute_log() {
        let mut ctx = Context::new();
        let expr = parse("(2*x + 1)^-1", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|2 * x + 1|) / (0 + 2)");
    }

    #[test]
    fn integrates_arctan_kernel() {
        let mut ctx = Context::new();
        let expr = parse("1/(x^2+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(x)");
    }

    #[test]
    fn integrates_polynomial_derivative_arctan_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arctan(x^2)");

        let expr = parse("x/(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arctan(x^2)");

        let expr = parse("2*x/(4+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arctan(x^2 / 2)");

        let expr = parse("1/(4+(x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arctan((x + 1) / 2)");
    }

    #[test]
    fn integrates_polynomial_derivative_atanh_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * atanh(x^2 / 2)");

        let expr = parse("x/(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/4 * atanh(x^2 / 2)");

        let expr = parse("1/(4-(x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * atanh((x + 1) / 2)");
    }

    #[test]
    fn integrates_polynomial_derivative_square_minus_constant_log_substitution() {
        let mut ctx = Context::new();
        let expr = parse("1/(x^2-1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * ln(|(x - 1) / (x + 1)|)");

        let expr = parse("2*x/(x^4-4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/4 * ln(|(x^2 - 2) / (x^2 + 2)|)");

        let expr = parse("x/(x^4-4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/8 * ln(|(x^2 - 2) / (x^2 + 2)|)");
    }

    #[test]
    fn integrates_polynomial_log_derivative_substitution() {
        let mut ctx = Context::new();
        let expr = parse("(2*x+1)/(x^2+x-1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|x^2 + x - 1|)");

        let expr = parse("(4*x+2)/(x^2+x+1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "2 * ln(|x^2 + x + 1|)");

        let expr = parse("(2*x+3)/((x+1)*(x+2))", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|(x + 1) * (x + 2)|)");
    }

    #[test]
    fn atanh_substitution_reports_open_interval_condition() {
        let mut ctx = Context::new();
        let expr = parse("2*x/(4-x^4)", &mut ctx).expect("parse");
        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");

        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "4 - x^4");

        let factored = parse("(2*x)/((2-x^2)*(x^2+2))", &mut ctx).expect("parse");
        let conditions =
            super::integrate_symbolic_required_positive_conditions(&mut ctx, factored, "x");

        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "(x^2 + 2) * (2 - x^2)");
    }

    #[test]
    fn integrates_polynomial_derivative_arcsin_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/sqrt(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arcsin(x^2 / 2)");

        let expr = parse("x/sqrt(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * arcsin(x^2 / 2)");

        let expr = parse("1/sqrt(4-(x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arcsin((x + 1) / 2)");

        let expr = parse("(2*x*sqrt(4-x^4))/(4-x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "arcsin(x^2 / 2)");
    }

    #[test]
    fn integrates_polynomial_derivative_asinh_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x/sqrt(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x^2)");

        let expr = parse("x/sqrt(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1/2 * asinh(x^2)");

        let expr = parse("2*x/sqrt(4+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x^2 / 2)");

        let expr = parse("1/sqrt(4+(x+1)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh((x + 1) / 2)");

        let expr = parse("(2*x*sqrt(1+x^4))/(1+x^4)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x^2)");
    }

    #[test]
    fn integrates_asinh_kernel() {
        let mut ctx = Context::new();
        let expr = parse("(x^2+1)^(-1/2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "asinh(x)");
    }

    #[test]
    fn integrates_secant_squared_kernel() {
        let mut ctx = Context::new();
        let expr = parse("1/cos(x)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sin(x) / cos(x)");

        let expr = parse("2*x/cos(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sin(x^2) / cos(x^2)");

        let expr = parse("x/cos(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "(sin(x^2) * 1/2)/cos(x^2)");
    }

    #[test]
    fn integrates_cosecant_squared_kernel() {
        let mut ctx = Context::new();
        let expr = parse("1/sin(x)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-(cos(x) / sin(x))");

        let expr = parse("3*x^2/sin(x^3)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-(cos(x^3) / sin(x^3))");
    }

    #[test]
    fn integrates_canonical_sec_tan_and_csc_cot_quotients() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x + 1)/cos(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "sec(2 * x + 1) / (0 + 2)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

        let expr = parse("cos(2*x + 1)/sin(2*x + 1)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-csc(2 * x + 1) / (0 + 2)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");
    }

    #[test]
    fn integrates_polynomial_sec_tan_and_csc_cot_quotients() {
        let mut ctx = Context::new();
        let expr = parse("2*x*sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1 / cos(x^2)");

        let expr = parse("x*sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "(1 * 1/2)/cos(x^2)");

        let expr = parse("3*x^2*cos(x^3)/sin(x^3)^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-(1 / sin(x^3))");

        let expr = parse("2*(x*sin(x^2)/cos(x^2)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "1 / cos(x^2)");

        let expr = parse("3*(x^2*cos(x^3)/sin(x^3)^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-(1 / sin(x^3))");
    }

    #[test]
    fn canonical_sec_tan_quotient_rejects_non_linear_argument() {
        let mut ctx = Context::new();
        let expr = parse("sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty()
        );
    }

    #[test]
    fn integrates_trig_log_linear_substitution() {
        let mut ctx = Context::new();
        let expr = parse("tan(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-ln(|cos(2 * x + 1)|) / (0 + 2)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

        let expr = parse("cot(2*x + 1)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sin(2 * x + 1)|) / (0 + 2)");

        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1);
        assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");
    }

    #[test]
    fn integrates_polynomial_trig_log_ratio_substitution() {
        let mut ctx = Context::new();
        let expr = parse("2*x*tan(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-ln(|cos(x^2)|)");

        let expr = parse("x*tan(x^2)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-1/2 * ln(|cos(x^2)|)");

        let expr = parse("3*x^2*cot(x^3)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "ln(|sin(x^3)|)");
    }

    #[test]
    fn trig_log_integration_rejects_non_linear_argument_without_condition() {
        let mut ctx = Context::new();
        let expr = parse("tan(x^2)", &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        assert!(
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty()
        );
    }

    #[test]
    fn extracts_linear_coeffs() {
        let mut ctx = Context::new();
        let expr = parse("2*x + 3", &mut ctx).expect("parse");
        let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("coeffs");
        let a_text = rendered(&ctx, a);
        assert!(a_text == "2" || a_text == "0 + 2");
        let b_text = rendered(&ctx, b);
        assert!(b_text == "3" || b_text == "0 + 3");
    }
}

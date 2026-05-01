//! Symbolic differentiation helpers shared by differentiation-facing rule layers.

use crate::build::mul2_raw;
use crate::expr_nary::{build_balanced_mul, mul_leaves};
use crate::expr_predicates::contains_named_var;
use crate::polynomial::Polynomial;
use crate::prove_sign::prove_positive_depth_with;
use crate::root_forms::{try_rewrite_simplify_square_root_expr, SimplifySquareRootRewriteKind};
use crate::tri_proof::TriProof;
use cas_ast::{ordering::compare_expr, BuiltinFn, Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

const SYMBOLIC_DIFF_SIGN_PROOF_DEPTH: usize = 8;

fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn add_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) {
        right
    } else if is_zero(ctx, right) {
        left
    } else {
        ctx.add(Expr::Add(left, right))
    }
}

fn sub_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, right) {
        left
    } else if is_zero(ctx, left) {
        ctx.add(Expr::Neg(right))
    } else {
        ctx.add(Expr::Sub(left, right))
    }
}

fn mul_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) || is_zero(ctx, right) {
        ctx.num(0)
    } else if is_one(ctx, left) {
        right
    } else if is_one(ctx, right) {
        left
    } else {
        mul2_raw(ctx, left, right)
    }
}

fn div_pruned(ctx: &mut Context, num: ExprId, den: ExprId) -> ExprId {
    if is_zero(ctx, num) {
        ctx.num(0)
    } else if is_one(ctx, den) {
        num
    } else {
        ctx.add(Expr::Div(num, den))
    }
}

fn hyperbolic_linear_factor(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    unary_builtin_arg(ctx, expr, BuiltinFn::Sinh)
        .map(|arg| (BuiltinFn::Sinh, arg))
        .or_else(|| unary_builtin_arg(ctx, expr, BuiltinFn::Cosh).map(|arg| (BuiltinFn::Cosh, arg)))
}

fn polynomial_times_builtin(ctx: &mut Context, poly: &Polynomial, builtin_expr: ExprId) -> ExprId {
    if poly.is_zero() {
        return ctx.num(0);
    }

    let poly_expr = poly.to_expr(ctx);
    mul_pruned(ctx, poly_expr, builtin_expr)
}

fn try_linear_times_hyperbolic_linear_derivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (hyperbolic_index, factor) in factors.iter().copied().enumerate() {
        let Some((builtin, arg)) = hyperbolic_linear_factor(ctx, factor) else {
            continue;
        };

        let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != hyperbolic_index).then_some(factor))
            .collect();
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if cofactor_poly.degree() > 1 {
            continue;
        }

        let cofactor_derivative = cofactor_poly.derivative();
        let slope_poly = Polynomial::new(vec![arg_slope], var.to_string());
        let scaled_cofactor = cofactor_poly.mul(&slope_poly);

        let companion_builtin = match builtin {
            BuiltinFn::Sinh => BuiltinFn::Cosh,
            BuiltinFn::Cosh => BuiltinFn::Sinh,
            _ => return None,
        };
        let companion = ctx.call_builtin(companion_builtin, vec![arg]);
        let term_from_cofactor = polynomial_times_builtin(ctx, &cofactor_derivative, factor);
        let term_from_chain = polynomial_times_builtin(ctx, &scaled_cofactor, companion);

        return Some(add_pruned(ctx, term_from_cofactor, term_from_chain));
    }

    None
}

fn one_minus_arg_square_sqrt(ctx: &mut Context, arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let inner = ctx.add(Expr::Sub(one, arg_sq));
    ctx.call_builtin(BuiltinFn::Sqrt, vec![inner])
}

fn positive_scaled_arg(ctx: &Context, arg: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(arg) {
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(scale) = positive_rational_const(ctx, l) {
                Some((scale, r))
            } else {
                positive_rational_const(ctx, r).map(|scale| (scale, l))
            }
        }
        Expr::Div(num, den) => {
            let den_value = positive_rational_const(ctx, *den)?;
            Some((reciprocal_positive_rational(&den_value), *num))
        }
        _ => None,
    }
}

fn positive_rational_const(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let value = cas_ast::views::as_rational_const(ctx, expr, 4)?;
    value.is_positive().then_some(value)
}

fn reciprocal_positive_rational(value: &BigRational) -> BigRational {
    BigRational::new(value.denom().clone(), value.numer().clone())
}

fn scaled_one_minus_arg_square_derivative(
    ctx: &mut Context,
    arg: ExprId,
    d_arg: ExprId,
) -> Option<ExprId> {
    let (scale, inner) = positive_scaled_arg(ctx, arg)?;
    if scale.is_one() {
        return None;
    }

    let scale_squared = &scale * &scale;
    let offset = BigRational::one() / scale_squared;
    let two = ctx.num(2);
    let inner_sq = ctx.add(Expr::Pow(inner, two));
    let offset_expr = ctx.add(Expr::Number(offset));
    let radicand = ctx.add(Expr::Sub(offset_expr, inner_sq));
    let den = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let numerator_scale = ctx.add(Expr::Number(reciprocal_positive_rational(&scale)));
    let numerator = mul_pruned(ctx, d_arg, numerator_scale);
    Some(ctx.add(Expr::Div(numerator, den)))
}

fn unit_reciprocal_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if is_one(ctx, *num) => Some(*den),
        Expr::Pow(base, exp)
            if matches!(
                ctx.get(*exp),
                Expr::Number(n) if n.is_integer() && n.to_integer() == (-1).into()
            ) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn one_minus_reciprocal_arg_square_sqrt(ctx: &mut Context, arg: ExprId) -> (ExprId, ExprId) {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let reciprocal_arg_sq = ctx.add(Expr::Div(one, arg_sq));
    let inner = ctx.add(Expr::Sub(one, reciprocal_arg_sq));
    (ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]), arg_sq)
}

fn is_intrinsically_positive_real(ctx: &Context, expr: ExprId) -> bool {
    prove_positive_depth_with(
        ctx,
        expr,
        SYMBOLIC_DIFF_SIGN_PROOF_DEPTH,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

fn is_positive_integer_power(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => match ctx.get(*exp) {
            Expr::Number(n) => n.is_integer() && n.is_positive(),
            _ => false,
        },
        _ => false,
    }
}

fn is_positive_even_integer_power(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => match ctx.get(*exp) {
            Expr::Number(n) => n.is_integer() && n.is_positive() && n.to_integer().is_even(),
            _ => false,
        },
        _ => false,
    }
}

fn acosh_left_radicand(ctx: &mut Context, arg: ExprId) -> ExprId {
    match ctx.get(arg) {
        Expr::Add(left, right) if is_one(ctx, *right) => *left,
        Expr::Add(left, right) if is_one(ctx, *left) => *right,
        _ => {
            let one = ctx.num(1);
            ctx.add(Expr::Sub(arg, one))
        }
    }
}

fn has_perfect_square_sqrt_rewrite(ctx: &mut Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {}
        _ => return false,
    }

    let sqrt_expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![expr]);
    try_rewrite_simplify_square_root_expr(ctx, sqrt_expr)
        .is_some_and(|rewrite| rewrite.kind == SimplifySquareRootRewriteKind::PerfectSquare)
}

fn positive_arg_arcsec_like_derivative(
    ctx: &mut Context,
    arg: ExprId,
    d_arg: ExprId,
    sign: i64,
) -> Option<ExprId> {
    if is_positive_integer_power(ctx, arg) || !is_intrinsically_positive_real(ctx, arg) {
        return None;
    }

    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let gap = ctx.add(Expr::Sub(arg_sq, one));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = mul_pruned(ctx, arg, sqrt_gap);
    let numerator = if sign < 0 {
        ctx.add(Expr::Neg(d_arg))
    } else {
        d_arg
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn arcsec_like_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId, sign: i64) -> ExprId {
    if let Some(derivative) = positive_arg_arcsec_like_derivative(ctx, arg, d_arg, sign) {
        return derivative;
    }

    let one = ctx.num(1);
    let (sqrt_gap, arg_sq) = one_minus_reciprocal_arg_square_sqrt(ctx, arg);
    let numerator = mul_pruned(ctx, d_arg, sqrt_gap);
    let numerator = if sign < 0 {
        ctx.add(Expr::Neg(numerator))
    } else {
        numerator
    };
    let denominator = ctx.add(Expr::Sub(arg_sq, one));
    ctx.add(Expr::Div(numerator, denominator))
}

fn arccot_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let denominator = ctx.add(Expr::Add(arg_sq, one));
    let numerator = ctx.add(Expr::Neg(d_arg));
    ctx.add(Expr::Div(numerator, denominator))
}

fn acosh_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let left = acosh_left_radicand(ctx, arg);
    let one = ctx.num(1);
    let minus_one = ctx.num(-1);
    let two = ctx.num(2);
    let neg_half = ctx.add(Expr::Div(minus_one, two));
    let right = ctx.add(Expr::Add(arg, one));

    if is_positive_even_integer_power(ctx, left) || has_perfect_square_sqrt_rewrite(ctx, left) {
        let sqrt_left = ctx.call_builtin(BuiltinFn::Sqrt, vec![left]);
        let sqrt_right = ctx.call_builtin(BuiltinFn::Sqrt, vec![right]);
        let denominator = mul_pruned(ctx, sqrt_left, sqrt_right);
        return div_pruned(ctx, d_arg, denominator);
    }

    let left_inv_sqrt = ctx.add(Expr::Pow(left, neg_half));
    let sqrt_right = ctx.call_builtin(BuiltinFn::Sqrt, vec![right]);
    let numerator = mul_pruned(ctx, d_arg, left_inv_sqrt);
    ctx.add(Expr::Div(numerator, sqrt_right))
}

fn atanh_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let minus_two = ctx.num(-2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let inner = ctx.add(Expr::Sub(one, arg_sq));
    let sqrt_inner = ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]);
    let inv_square = ctx.add(Expr::Pow(sqrt_inner, minus_two));
    mul_pruned(ctx, d_arg, inv_square)
}

fn constant_base_log_derivative(
    ctx: &mut Context,
    arg: ExprId,
    d_arg: ExprId,
    base: i64,
) -> ExprId {
    let base = ctx.num(base);
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let den = mul_pruned(ctx, arg, ln_base);
    ctx.add(Expr::Div(d_arg, den))
}

#[derive(Clone, Copy)]
enum ReciprocalTrigDerivativeKind {
    Sec,
    Csc,
    Cot,
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(builtin) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn canonical_reciprocal_trig_div_kind(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ReciprocalTrigDerivativeKind, ExprId)> {
    if is_one(ctx, num) {
        if let Some(arg) = unary_builtin_arg(ctx, den, BuiltinFn::Cos) {
            return Some((ReciprocalTrigDerivativeKind::Sec, arg));
        }
        if let Some(arg) = unary_builtin_arg(ctx, den, BuiltinFn::Sin) {
            return Some((ReciprocalTrigDerivativeKind::Csc, arg));
        }
    }

    let cos_arg = unary_builtin_arg(ctx, num, BuiltinFn::Cos)?;
    let sin_arg = unary_builtin_arg(ctx, den, BuiltinFn::Sin)?;
    if compare_expr(ctx, cos_arg, sin_arg) == Ordering::Equal {
        Some((ReciprocalTrigDerivativeKind::Cot, cos_arg))
    } else {
        None
    }
}

fn canonical_hyperbolic_coth_div_arg(ctx: &Context, num: ExprId, den: ExprId) -> Option<ExprId> {
    let cosh_arg = unary_builtin_arg(ctx, num, BuiltinFn::Cosh)?;
    let sinh_arg = unary_builtin_arg(ctx, den, BuiltinFn::Sinh)?;
    (compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal).then_some(cosh_arg)
}

fn squared_builtin_call(ctx: &mut Context, builtin: BuiltinFn, arg: ExprId) -> ExprId {
    let call = ctx.call_builtin(builtin, vec![arg]);
    let two = ctx.num(2);
    ctx.add(Expr::Pow(call, two))
}

fn secant_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_sq = squared_builtin_call(ctx, BuiltinFn::Cos, arg);
    let numerator = mul_pruned(ctx, sin_u, d_arg);
    ctx.add(Expr::Div(numerator, cos_sq))
}

fn cosecant_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let neg_cos = ctx.add(Expr::Neg(cos_u));
    let sin_sq = squared_builtin_call(ctx, BuiltinFn::Sin, arg);
    let numerator = mul_pruned(ctx, neg_cos, d_arg);
    ctx.add(Expr::Div(numerator, sin_sq))
}

fn cotangent_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let sin_sq = squared_builtin_call(ctx, BuiltinFn::Sin, arg);
    let neg_d_arg = ctx.add(Expr::Neg(d_arg));
    ctx.add(Expr::Div(neg_d_arg, sin_sq))
}

fn hyperbolic_coth_derivative(ctx: &mut Context, arg: ExprId, d_arg: ExprId) -> ExprId {
    let sinh_sq = squared_builtin_call(ctx, BuiltinFn::Sinh, arg);
    let neg_d_arg = ctx.add(Expr::Neg(d_arg));
    ctx.add(Expr::Div(neg_d_arg, sinh_sq))
}

fn reciprocal_trig_derivative(
    ctx: &mut Context,
    kind: ReciprocalTrigDerivativeKind,
    arg: ExprId,
    d_arg: ExprId,
) -> ExprId {
    match kind {
        ReciprocalTrigDerivativeKind::Sec => secant_derivative(ctx, arg, d_arg),
        ReciprocalTrigDerivativeKind::Csc => cosecant_derivative(ctx, arg, d_arg),
        ReciprocalTrigDerivativeKind::Cot => cotangent_derivative(ctx, arg, d_arg),
    }
}

fn log_abs_builtin_derivative(ctx: &mut Context, arg: ExprId, var: &str) -> Option<ExprId> {
    let inner_expr = unary_builtin_arg(ctx, arg, BuiltinFn::Abs)?;

    if let Expr::Div(num, den) = ctx.get(inner_expr) {
        let (num, den) = (*num, *den);
        let d_num = differentiate_symbolic_expr(ctx, num, var)?;
        let d_den = differentiate_symbolic_expr(ctx, den, var)?;
        if is_zero(ctx, d_den) {
            return Some(div_pruned(ctx, d_num, num));
        }
        if is_zero(ctx, d_num) {
            let den_part = div_pruned(ctx, d_den, den);
            return Some(ctx.add(Expr::Neg(den_part)));
        }

        let term1 = mul_pruned(ctx, d_num, den);
        let term2 = mul_pruned(ctx, num, d_den);
        let numerator = sub_pruned(ctx, term1, term2);
        let denominator = mul_pruned(ctx, num, den);
        return Some(div_pruned(ctx, numerator, denominator));
    }

    if let Expr::Mul(left, right) = ctx.get(inner_expr) {
        let (left, right) = (*left, *right);
        let d_left = differentiate_symbolic_expr(ctx, left, var)?;
        let d_right = differentiate_symbolic_expr(ctx, right, var)?;
        if is_zero(ctx, d_right) {
            return Some(div_pruned(ctx, d_left, left));
        }
        if is_zero(ctx, d_left) {
            return Some(div_pruned(ctx, d_right, right));
        }

        let term1 = mul_pruned(ctx, d_left, right);
        let term2 = mul_pruned(ctx, left, d_right);
        let numerator = add_pruned(ctx, term1, term2);
        let denominator = mul_pruned(ctx, left, right);
        return Some(div_pruned(ctx, numerator, denominator));
    }

    if let Some(trig_arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Sin) {
        let d_trig_arg = differentiate_symbolic_expr(ctx, trig_arg, var)?;
        let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![trig_arg]);
        let numerator = mul_pruned(ctx, d_trig_arg, cos_u);
        return Some(ctx.add(Expr::Div(numerator, inner_expr)));
    }

    if let Some(trig_arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Cos) {
        let d_trig_arg = differentiate_symbolic_expr(ctx, trig_arg, var)?;
        let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![trig_arg]);
        let neg_sin_u = ctx.add(Expr::Neg(sin_u));
        let numerator = mul_pruned(ctx, d_trig_arg, neg_sin_u);
        return Some(ctx.add(Expr::Div(numerator, inner_expr)));
    }

    if let Some(hyperbolic_arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Sinh) {
        let d_hyperbolic_arg = differentiate_symbolic_expr(ctx, hyperbolic_arg, var)?;
        let tanh_u = ctx.call_builtin(BuiltinFn::Tanh, vec![hyperbolic_arg]);
        return Some(ctx.add(Expr::Div(d_hyperbolic_arg, tanh_u)));
    }

    if let Some(hyperbolic_arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Cosh) {
        let d_hyperbolic_arg = differentiate_symbolic_expr(ctx, hyperbolic_arg, var)?;
        let tanh_u = ctx.call_builtin(BuiltinFn::Tanh, vec![hyperbolic_arg]);
        return Some(mul_pruned(ctx, d_hyperbolic_arg, tanh_u));
    }

    let d_inner = differentiate_symbolic_expr(ctx, inner_expr, var)?;
    Some(div_pruned(ctx, d_inner, inner_expr))
}

fn fixed_base_log_abs_derivative(
    ctx: &mut Context,
    arg: ExprId,
    base: ExprId,
    var: &str,
) -> Option<ExprId> {
    let derivative = log_abs_builtin_derivative(ctx, arg, var)?;
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let one = ctx.num(1);
    let reciprocal_ln_base = ctx.add(Expr::Div(one, ln_base));
    Some(mul_pruned(ctx, reciprocal_ln_base, derivative))
}

fn variable_base_log_abs_derivative(
    ctx: &mut Context,
    base: ExprId,
    d_base: ExprId,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let arg_ratio = log_abs_builtin_derivative(ctx, arg, var)?;
    let ln_arg = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let two = ctx.num(2);
    let ln_base_sq = ctx.add(Expr::Pow(ln_base, two));
    let base_ratio =
        log_abs_builtin_derivative(ctx, base, var).unwrap_or_else(|| div_pruned(ctx, d_base, base));
    let term_arg = mul_pruned(ctx, arg_ratio, ln_base);
    let term_base = mul_pruned(ctx, ln_arg, base_ratio);
    let numerator = sub_pruned(ctx, term_arg, term_base);
    Some(ctx.add(Expr::Div(numerator, ln_base_sq)))
}

fn numeric_fixed_base_log_abs_derivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
    base: i64,
) -> Option<ExprId> {
    let base = ctx.num(base);
    fixed_base_log_abs_derivative(ctx, arg, base, var)
}

fn abs_quotient_derivative(
    ctx: &mut Context,
    arg: ExprId,
    quotient_num: ExprId,
    quotient_den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let d_num = differentiate_symbolic_expr(ctx, quotient_num, var)?;
    let d_den = differentiate_symbolic_expr(ctx, quotient_den, var)?;
    let term1 = mul_pruned(ctx, d_num, quotient_den);
    let term2 = mul_pruned(ctx, quotient_num, d_den);
    let quotient_derivative_num = sub_pruned(ctx, term1, term2);
    let numerator = mul_pruned(ctx, quotient_num, quotient_derivative_num);

    let three = ctx.num(3);
    let den_cubed = ctx.add(Expr::Pow(quotient_den, three));
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    let denominator = mul_pruned(ctx, abs_arg, den_cubed);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

/// Differentiate `expr` with respect to variable `var`.
pub fn differentiate_symbolic_expr(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    if !contains_named_var(ctx, expr, var) {
        return Some(ctx.num(0));
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) => {
            if ctx.sym_name(*sym_id) == var {
                Some(ctx.num(1))
            } else {
                Some(ctx.num(0))
            }
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(add_pruned(ctx, dl, dr))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(sub_pruned(ctx, dl, dr))
        }
        Expr::Neg(inner) => {
            let inner = *inner;
            let d_inner = differentiate_symbolic_expr(ctx, inner, var)?;
            Some(ctx.add(Expr::Neg(d_inner)))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(derivative) = try_linear_times_hyperbolic_linear_derivative(ctx, expr, var)
            {
                return Some(derivative);
            }
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul_pruned(ctx, dl, r);
            let term2 = mul_pruned(ctx, l, dr);
            Some(add_pruned(ctx, term1, term2))
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            if let Some((kind, arg)) = canonical_reciprocal_trig_div_kind(ctx, l, r) {
                let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
                return Some(reciprocal_trig_derivative(ctx, kind, arg, d_arg));
            }
            if let Some(arg) = canonical_hyperbolic_coth_div_arg(ctx, l, r) {
                let d_arg = differentiate_symbolic_expr(ctx, arg, var)?;
                return Some(hyperbolic_coth_derivative(ctx, arg, d_arg));
            }

            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul_pruned(ctx, dl, r);
            let term2 = mul_pruned(ctx, l, dr);
            let num = sub_pruned(ctx, term1, term2);
            let two = ctx.num(2);
            let den = ctx.add(Expr::Pow(r, two));
            Some(ctx.add(Expr::Div(num, den)))
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            let db = differentiate_symbolic_expr(ctx, base, var)?;
            let de = differentiate_symbolic_expr(ctx, exp, var)?;

            if !contains_named_var(ctx, exp, var) {
                let one = ctx.num(1);
                let n_minus_one = ctx.add(Expr::Sub(exp, one));
                let pow_term = ctx.add(Expr::Pow(base, n_minus_one));
                let term = mul_pruned(ctx, exp, pow_term);
                Some(mul_pruned(ctx, term, db))
            } else if !contains_named_var(ctx, base, var) {
                let ln_a = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term = mul_pruned(ctx, expr, ln_a);
                Some(mul_pruned(ctx, term, de))
            } else {
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term1 = mul_pruned(ctx, de, ln_base);
                let term2_num = mul_pruned(ctx, exp, db);
                let term2 = ctx.add(Expr::Div(term2_num, base));
                let inner = add_pruned(ctx, term1, term2);
                Some(mul_pruned(ctx, expr, inner))
            }
        }
        Expr::Function(fn_id, args) => {
            let (fn_id, args) = (*fn_id, args.clone());
            if matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Log)) && args.len() == 2 {
                let base = args[0];
                let arg = args[1];
                if contains_named_var(ctx, base, var) {
                    let db = differentiate_symbolic_expr(ctx, base, var)?;
                    let ln_arg = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
                    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                    let two = ctx.num(2);
                    let ln_base_sq = ctx.add(Expr::Pow(ln_base, two));

                    if contains_named_var(ctx, arg, var) {
                        if let Some(derivative) =
                            variable_base_log_abs_derivative(ctx, base, db, arg, var)
                        {
                            return Some(derivative);
                        }

                        let da = differentiate_symbolic_expr(ctx, arg, var)?;
                        let arg_ratio = ctx.add(Expr::Div(da, arg));
                        let base_ratio = ctx.add(Expr::Div(db, base));
                        let term_arg = mul_pruned(ctx, arg_ratio, ln_base);
                        let term_base = mul_pruned(ctx, ln_arg, base_ratio);
                        let numerator = sub_pruned(ctx, term_arg, term_base);
                        return Some(ctx.add(Expr::Div(numerator, ln_base_sq)));
                    }

                    let denominator = mul_pruned(ctx, base, ln_base_sq);
                    let numerator = mul_pruned(ctx, ln_arg, db);
                    let neg_numerator = ctx.add(Expr::Neg(numerator));
                    return Some(ctx.add(Expr::Div(neg_numerator, denominator)));
                }

                if let Some(derivative) = fixed_base_log_abs_derivative(ctx, arg, base, var) {
                    return Some(derivative);
                }

                let da = differentiate_symbolic_expr(ctx, arg, var)?;
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let den = mul_pruned(ctx, arg, ln_base);
                return Some(ctx.add(Expr::Div(da, den)));
            }

            if args.len() != 1 {
                return None;
            }
            let arg = args[0];

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) {
                if let Some(derivative) = log_abs_builtin_derivative(ctx, arg, var) {
                    return Some(derivative);
                }
            }
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Log2) {
                if let Some(derivative) = numeric_fixed_base_log_abs_derivative(ctx, arg, var, 2) {
                    return Some(derivative);
                }
            }
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Log10) {
                if let Some(derivative) = numeric_fixed_base_log_abs_derivative(ctx, arg, var, 10) {
                    return Some(derivative);
                }
            }

            if let Some(recip_base) = unit_reciprocal_base(ctx, arg) {
                match ctx.builtin_of(fn_id) {
                    Some(BuiltinFn::Arccos | BuiltinFn::Acos) => {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arcsec_like_derivative(ctx, recip_base, d_base, 1));
                    }
                    Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arcsec_like_derivative(ctx, recip_base, d_base, -1));
                    }
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
                        let d_base = differentiate_symbolic_expr(ctx, recip_base, var)?;
                        return Some(arccot_derivative(ctx, recip_base, d_base));
                    }
                    _ => {}
                }
            }

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Abs) {
                if let Expr::Div(num, den) = ctx.get(arg) {
                    let (num, den) = (*num, *den);
                    return abs_quotient_derivative(ctx, arg, num, den, var);
                }
            }

            let da = differentiate_symbolic_expr(ctx, arg, var)?;

            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Sin) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    Some(mul_pruned(ctx, cos_u, da))
                }
                Some(BuiltinFn::Cos) => {
                    let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                    let neg_sin = ctx.add(Expr::Neg(sin_u));
                    Some(mul_pruned(ctx, neg_sin, da))
                }
                Some(BuiltinFn::Tan) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    let two = ctx.num(2);
                    let cos_sq = ctx.add(Expr::Pow(cos_u, two));
                    let one = ctx.num(1);
                    let sec_sq = ctx.add(Expr::Div(one, cos_sq));
                    Some(mul_pruned(ctx, sec_sq, da))
                }
                Some(BuiltinFn::Sec) => Some(secant_derivative(ctx, arg, da)),
                Some(BuiltinFn::Csc) => Some(cosecant_derivative(ctx, arg, da)),
                Some(BuiltinFn::Cot) => Some(cotangent_derivative(ctx, arg, da)),
                Some(BuiltinFn::Sinh) => {
                    let cosh_u = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                    Some(mul_pruned(ctx, cosh_u, da))
                }
                Some(BuiltinFn::Cosh) => {
                    let sinh_u = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                    Some(mul_pruned(ctx, sinh_u, da))
                }
                Some(BuiltinFn::Tanh) => {
                    let cosh_u = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                    let two = ctx.num(2);
                    let cosh_sq = ctx.add(Expr::Pow(cosh_u, two));
                    let one = ctx.num(1);
                    let sech_sq = ctx.add(Expr::Div(one, cosh_sq));
                    Some(mul_pruned(ctx, sech_sq, da))
                }
                Some(BuiltinFn::Sqrt) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let half = ctx.add(Expr::Div(one, two));
                    let minus_one = ctx.num(-1);
                    let neg_half = ctx.add(Expr::Div(minus_one, two));
                    let pow_term = ctx.add(Expr::Pow(arg, neg_half));
                    let term = mul_pruned(ctx, half, pow_term);
                    Some(mul_pruned(ctx, term, da))
                }
                Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let arg_sq = ctx.add(Expr::Pow(arg, two));
                    let den = ctx.add(Expr::Add(one, arg_sq));
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => {
                    if let Some(scaled) = scaled_one_minus_arg_square_derivative(ctx, arg, da) {
                        return Some(scaled);
                    }
                    let den = one_minus_arg_square_sqrt(ctx, arg);
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Arccos | BuiltinFn::Acos) => {
                    let neg_da = ctx.add(Expr::Neg(da));
                    if let Some(scaled) = scaled_one_minus_arg_square_derivative(ctx, arg, neg_da) {
                        return Some(scaled);
                    }
                    let den = one_minus_arg_square_sqrt(ctx, arg);
                    Some(ctx.add(Expr::Div(neg_da, den)))
                }
                Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => {
                    Some(arcsec_like_derivative(ctx, arg, da, 1))
                }
                Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => {
                    Some(arcsec_like_derivative(ctx, arg, da, -1))
                }
                Some(BuiltinFn::Arccot | BuiltinFn::Acot) => Some(arccot_derivative(ctx, arg, da)),
                Some(BuiltinFn::Asinh) => {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let arg_sq = ctx.add(Expr::Pow(arg, two));
                    let inner = ctx.add(Expr::Add(arg_sq, one));
                    let den = ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]);
                    Some(ctx.add(Expr::Div(da, den)))
                }
                Some(BuiltinFn::Acosh) => Some(acosh_derivative(ctx, arg, da)),
                Some(BuiltinFn::Atanh) => Some(atanh_derivative(ctx, arg, da)),
                Some(BuiltinFn::Exp) => Some(mul_pruned(ctx, expr, da)),
                Some(BuiltinFn::Ln) => Some(ctx.add(Expr::Div(da, arg))),
                Some(BuiltinFn::Log2) => Some(constant_base_log_derivative(ctx, arg, da, 2)),
                Some(BuiltinFn::Log10) => Some(constant_base_log_derivative(ctx, arg, da, 10)),
                Some(BuiltinFn::Abs) => {
                    let term = ctx.add(Expr::Div(arg, expr));
                    Some(mul_pruned(ctx, term, da))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::differentiate_symbolic_expr;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn differentiates_product_rule() {
        let mut ctx = Context::new();
        let expr = parse("x*sin(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("sin(x)"));
        assert!(text.contains("cos(x)"));
    }

    #[test]
    fn differentiates_chain_rule_exp() {
        let mut ctx = Context::new();
        let expr = parse("exp(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("exp(x^2)") || text.contains("e^(x^2)"));
    }

    #[test]
    fn prunes_zero_product_terms_from_polynomial_derivative() {
        let mut ctx = Context::new();
        let expr = parse("x^3 + 2*x^2 - 5*x + 1", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(!text.contains("0 ·"), "unexpected zero product in {text}");
        assert!(!text.contains("· 1"), "unexpected unit factor in {text}");
        assert!(text.contains("3"));
        assert!(text.contains("- 5"));
    }

    #[test]
    fn prunes_unit_chain_factor_for_sine() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "cos(x)");
    }

    #[test]
    fn differentiates_reciprocal_trig_functions_directly() {
        let cases = [
            ("sec(x)", "sin(x) / cos(x)^2"),
            ("csc(x)", "-cos(x) / sin(x)^2"),
            ("cot(x)", "-1 / sin(x)^2"),
            ("1/cos(x)", "sin(x) / cos(x)^2"),
            ("1/sin(x)", "-cos(x) / sin(x)^2"),
            ("cos(x)/sin(x)", "-1 / sin(x)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }

        let mut ctx = Context::new();
        let expr = parse("acosh(x^2 + 2*x + 2)", &mut ctx).expect("parse expanded square");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff expanded square");
        let text = rendered(&ctx, out);

        assert!(
            text.contains("sqrt(x^2 + 2 * x + 2 - 1)"),
            "expanded perfect-square acosh derivative should keep sqrt(left): {text}"
        );
        assert!(
            !text.contains("(x^2 + 2 * x + 2 - 1)^(-1 / 2)"),
            "expanded perfect-square acosh derivative should avoid inverse-power left radicand: {text}"
        );
    }

    #[test]
    fn differentiates_reciprocal_trig_chain_rule_directly() {
        let cases = [
            ("sec(2*x + 1)", "sin(2 * x + 1) * 2 / cos(2 * x + 1)^2"),
            ("csc(2*x + 1)", "-cos(2 * x + 1) * 2 / sin(2 * x + 1)^2"),
            ("cot(2*x + 1)", "-2 / sin(2 * x + 1)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_log_abs_builtins_through_domain_carrying_quotients() {
        let cases = [
            ("ln(abs(sin(x)))", "cos(x) / sin(x)"),
            ("ln(abs(cos(x)))", "-sin(x) / cos(x)"),
            ("ln(abs(sinh(x)))", "1 / tanh(x)"),
            ("ln(abs(cosh(x)))", "tanh(x)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_log_abs_quotient_without_abs_squared_noise() {
        let cases = [
            (
                "ln(abs((x-1)/(x+1)))",
                "(x + 1 - (x - 1)) / ((x - 1) * (x + 1))",
            ),
            ("ln(abs(x/y))", "1 / x"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_log_abs_product_without_abs_squared_noise() {
        let cases = [
            ("ln(abs(x*y))", "1 / x"),
            (
                "ln(abs((x-1)*(x+1)))",
                "(x + x + 1 - 1) / ((x + 1) * (x - 1))",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_generic_log_abs_without_abs_squared_noise() {
        let mut ctx = Context::new();
        let expr = parse("ln(abs(x^2-1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(
            text.contains("/ (x^2 - 1)") || text.contains("/(x^2 - 1)"),
            "generic ln(abs(u)) derivative should divide by u directly: {text}"
        );
        assert!(
            !text.contains('|'),
            "generic ln(abs(u)) derivative should not route through abs noise: {text}"
        );
    }

    #[test]
    fn differentiates_fixed_base_log_abs_without_abs_squared_noise() {
        let cases = [
            ("log(2, abs(x^2-1))", "ln(2)"),
            ("log(y, abs(x^2-1))", "ln(y)"),
            ("log2(abs(x^2-1))", "ln(2)"),
            ("log10(abs(x^2-1))", "ln(10)"),
        ];

        for (input, expected_log_base) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                text.contains(expected_log_base),
                "fixed-base log(abs(u)) derivative should divide by {expected_log_base}: {text}"
            );
            assert!(
                text.contains("x^2 - 1"),
                "fixed-base log(abs(u)) derivative should divide by u directly: {text}"
            );
            assert!(
                !text.contains('|'),
                "fixed-base log(abs(u)) derivative should not route through abs noise: {text}"
            );
        }
    }

    #[test]
    fn differentiates_variable_base_log_abs_without_abs_squared_noise() {
        let mut ctx = Context::new();
        let expr = parse("log(x, abs(x^2-1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("ln(x)"), "{text}");
        assert!(text.contains("ln(|x^2 - 1|)"), "{text}");
        assert!(
            text.contains("/ (x^2 - 1)")
                || text.contains("/(x^2 - 1)")
                || text.contains("/((x^2 - 1))"),
            "variable-base log(abs(u)) derivative should divide by u directly: {text}"
        );
        assert!(
            !text.contains("/ |x^2 - 1|") && !text.contains("/(|x^2 - 1|"),
            "variable-base log(abs(u)) derivative should not divide by abs(u): {text}"
        );
        assert!(
            !text.contains("|x^2 - 1|)^2"),
            "variable-base log(abs(u)) derivative should not route through abs-squared cleanup: {text}"
        );
    }

    #[test]
    fn differentiates_variable_abs_base_log_abs_without_abs_base_noise() {
        let mut ctx = Context::new();
        let expr = parse("log(abs(x), abs(x^2-1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("ln(|x|)"), "{text}");
        assert!(text.contains("ln(|x^2 - 1|)"), "{text}");
        assert!(
            text.contains("/ x") || text.contains("/x"),
            "variable abs-base log(abs(u)) derivative should divide by the base inner directly: {text}"
        );
        assert!(
            !text.contains("/ |x|") && !text.contains("x / |x|"),
            "variable abs-base log(abs(u)) derivative should not route through abs-base cleanup: {text}"
        );
    }

    #[test]
    fn differentiates_abs_quotient_without_nested_fraction_noise() {
        let mut ctx = Context::new();
        let expr = parse("abs((x-1)/(x+1))", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("|(x - 1) / (x + 1)|"), "{text}");
        assert!(text.contains("(x + 1)^3"), "{text}");
        assert!(
            !text.contains("(x + 1)^2"),
            "unexpected quotient-rule denominator expansion in {text}"
        );
    }

    #[test]
    fn differentiates_hyperbolic_functions() {
        let cases = [
            ("sinh(x)", "cosh(x)"),
            ("cosh(x)", "sinh(x)"),
            ("tanh(x)", "1 / cosh(x)^2"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_hyperbolic_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("sinh(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("cosh(x^2)"), "{text}");
        assert!(text.contains("2 * x"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_linear_times_hyperbolic_linear_with_compact_chain_factor() {
        let cases = [
            (
                "(2/3*x+2/3)*cosh((3*x+2)/2)",
                "2/3 * cosh((3 * x + 2) / 2) + (x + 1) * sinh((3 * x + 2) / 2)",
            ),
            ("4/9*sinh((3*x+2)/2)", "2/3 * cosh((3 * x + 2) / 2)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_hyperbolic_coth_quotient_directly() {
        let mut ctx = Context::new();
        let expr = parse("cosh(x)/sinh(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

        assert_eq!(rendered(&ctx, out), "-1 / sinh(x)^2");
    }

    #[test]
    fn differentiates_total_domain_inverse_functions() {
        let cases = [("arctan(x)", "x^2 + 1"), ("asinh(x)", "x^2 + 1")];

        for (input, expected_core) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(text.contains(expected_core), "input: {input}, got: {text}");
            assert!(!text.contains("diff("), "input: {input}, got: {text}");
        }
    }

    #[test]
    fn differentiates_constant_base_unary_logs() {
        let cases = [
            ("log2(x)", "1 / (x * ln(2))"),
            ("log10(x)", "1 / (x * ln(10))"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_variable_base_constant_argument_logs_conservatively() {
        let cases = [("log(x, 2)", "ln(2)"), ("log(x, y)", "ln(y)")];

        for (input, expected_log_arg) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                text.contains(expected_log_arg),
                "input: {input}, got: {text}"
            );
            assert!(text.contains("ln(x)^2"), "input: {input}, got: {text}");
            assert!(!text.contains("diff("), "input: {input}, got: {text}");
        }

        let mut ctx = Context::new();
        let expr = parse("log(x, x + 1)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("ln(x)"), "got: {text}");
        assert!(text.contains("ln(x + 1)"), "got: {text}");
        assert!(text.contains("ln(x)^2"), "got: {text}");
        assert!(!text.contains("diff("), "got: {text}");
    }

    #[test]
    fn differentiates_inverse_reciprocal_trig_directly() {
        let cases = [
            ("arcsec(x)", "sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("arccsc(x)", "-sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("arccot(x)", "-1 / (x^2 + 1)"),
            ("asec(x)", "sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("acsc(x)", "-sqrt(1 - 1 / x^2) / (x^2 - 1)"),
            ("acot(x)", "-1 / (x^2 + 1)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn inverse_reciprocal_trig_chain_rule_keeps_compact_core() {
        let mut ctx = Context::new();
        let expr = parse("arcsec((x^2 + 1)^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("sqrt(1 - 1 / (x^2 + 1)^2^2)"), "{text}");
        assert!(
            !text.contains("+") || !text.contains("x^3"),
            "unexpected quotient-rule expansion in {text}"
        );
    }

    #[test]
    fn positive_inverse_reciprocal_trig_chain_rule_uses_direct_positive_argument_form() {
        let cases = [("arcsec(x^2 + 1)", false), ("arccsc(x^2 + 1)", true)];

        for (input, expect_negative) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                !text.contains("1 - 1 /"),
                "positive argument derivative should not use reciprocal-square gap: {text}"
            );
            assert!(
                text.contains("sqrt((x^2 + 1)^2 - 1)"),
                "positive argument derivative should expose the direct gap: {text}"
            );
            assert_eq!(
                text.starts_with("-"),
                expect_negative,
                "unexpected sign for {input}: {text}"
            );
        }
    }

    #[test]
    fn reciprocal_inverse_trig_rewrite_targets_keep_compact_derivative_core() {
        let cases = [
            "arccos(1/(x^2 + 1)^2)",
            "arcsin(1/(x^2 + 1)^2)",
            "arctan(1/(x^2 + 1)^2)",
        ];

        for input in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
            let text = rendered(&ctx, out);

            assert!(
                !text.contains("/sqrt("),
                "unexpected quotient-rule reciprocal derivative in {text}"
            );
            assert!(
                !text.contains("x^3"),
                "unexpected expanded chain factor in {text}"
            );
        }
    }

    #[test]
    fn differentiates_acosh_with_domain_safe_radicals() {
        let cases = [
            ("acosh(x)", "(x - 1)^(-1 / 2) / sqrt(x + 1)"),
            (
                "acosh(2*x + 1)",
                "2 * (2 * x)^(-1 / 2) / sqrt(2 * x + 1 + 1)",
            ),
            (
                "acosh(x^2 + 1)",
                "2 * x^(2 - 1) / (sqrt(x^2) * sqrt(x^2 + 1 + 1))",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_atanh_with_open_unit_interval_witness() {
        let cases = [
            ("atanh(x)", "sqrt(1 - x^2)^(-2)"),
            ("atanh(x^2)", "(x^(2 - 1) * 2)/(sqrt(1 - x^2^2)^2)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_sqrt_chain_rule_without_power_presimplification() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(2 - x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("(2 - x)^(-1 / 2)"), "{text}");
        assert!(text.contains("-1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_power_chain_rule_through_canonical_negation() {
        let mut ctx = Context::new();
        let expr = parse("(2 + (-x))^(1/2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("(2 - x)^(1 / 2 - 1)"), "{text}");
        assert!(text.contains("-1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_bounded_inverse_trig_functions() {
        let cases = [
            ("arcsin(x)", "1 / sqrt(1 - x^2)"),
            ("arccos(x)", "-1 / sqrt(1 - x^2)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).expect("parse");
            let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");

            assert_eq!(rendered(&ctx, out), expected, "input: {input}");
        }
    }

    #[test]
    fn differentiates_bounded_inverse_trig_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("asin(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("2 * x"), "{text}");
        assert!(text.contains("sqrt(1 - x^2^2)"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_inverse_function_chain_rule() {
        let mut ctx = Context::new();
        let expr = parse("arctan(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(text.contains("2 * x"), "{text}");
        assert!(text.contains("x^2^2 + 1"), "{text}");
        assert!(!text.contains("diff("), "{text}");
    }

    #[test]
    fn differentiates_log_with_constant_base() {
        let mut ctx = Context::new();
        let expr = parse("log(2, x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);

        assert!(
            text.contains("x * ln(2)") || text.contains("ln(2) * x"),
            "{text}"
        );
        assert!(!text.contains("diff("), "{text}");
    }
}

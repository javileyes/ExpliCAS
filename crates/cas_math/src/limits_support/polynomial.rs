//! `limits_support`: familia `polynomial`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

pub(super) fn apply_finite_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();

    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    let value = poly.eval(&point_value);
    Some(ctx.add(Expr::Number(value)))
}

pub(super) fn apply_finite_one_sided_sign_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Sign) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    let (order, derivative) =
        finite_polynomial_local_order_and_derivative(&argument, &point_value)?;
    let tail_sign = finite_local_tail_sign(&derivative, order, side)?;
    Some(signed_unit_limit(ctx, tail_sign))
}

pub(super) fn apply_finite_bilateral_sign_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left =
        apply_finite_one_sided_sign_polynomial_rule(ctx, expr, var, point, FiniteLimitSide::Left)?;
    let right =
        apply_finite_one_sided_sign_polynomial_rule(ctx, expr, var, point, FiniteLimitSide::Right)?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

pub(super) fn apply_finite_acosh_polynomial_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], &var_name).ok()?;
    if argument.eval(&point_value) != rational_one() {
        return None;
    }

    let endpoint_gap = argument.sub(&Polynomial::one(var_name));
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&endpoint_gap, &point_value)?;
    if !finite_local_tail_positive_on_both_sides(&gap_derivative, gap_order)? {
        return None;
    }

    Some(ctx.num(0))
}

/// Recognize `scale * (a^(g) - 1)` with `a` ANY power base (callers gate the
/// base — rational, or provable-constant like `π`); returns `(scale, a, g)`.
/// Handles the Sub and Add(-1) offset forms, a numeric scale factor, and a
/// negation.
pub(super) fn scaled_general_power_zero_offset(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId, ExprId)> {
    let power_parts = |ctx: &Context, e: ExprId| -> Option<(ExprId, ExprId)> {
        match ctx.get(e) {
            Expr::Pow(base, exponent) => Some((*base, *exponent)),
            _ => None,
        }
    };
    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if expr_is_one(ctx, rhs) => {
            let (base, exponent) = power_parts(ctx, lhs)?;
            Some((rational_one(), base, exponent))
        }
        // 1 - a^g = -(a^g - 1).
        Expr::Sub(lhs, rhs) if expr_is_one(ctx, lhs) => {
            let (base, exponent) = power_parts(ctx, rhs)?;
            Some((-rational_one(), base, exponent))
        }
        Expr::Add(lhs, rhs) => {
            if constant_rational_value(ctx, rhs).is_some_and(|v| v == -rational_one()) {
                let (base, exponent) = power_parts(ctx, lhs)?;
                return Some((rational_one(), base, exponent));
            }
            if constant_rational_value(ctx, lhs).is_some_and(|v| v == -rational_one()) {
                let (base, exponent) = power_parts(ctx, rhs)?;
                return Some((rational_one(), base, exponent));
            }
            None
        }
        Expr::Neg(inner) => {
            let (scale, base, exponent) = scaled_general_power_zero_offset(ctx, inner)?;
            Some((-scale, base, exponent))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, base, exponent) = scaled_general_power_zero_offset(ctx, rhs)?;
                return Some((scale * inner_scale, base, exponent));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, base, exponent) = scaled_general_power_zero_offset(ctx, lhs)?;
                return Some((scale * inner_scale, base, exponent));
            }
            None
        }
        _ => None,
    }
}

/// `a^(exponent)` with `a` a numeric rational base; returns `(a, exponent)`.
pub(super) fn numeric_base_power(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            let base_value = constant_rational_value(ctx, *base)?;
            Some((base_value, *exponent))
        }
        _ => None,
    }
}

fn finite_sqrt_even_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if exponent % 2 != 0 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, base_limit)?;
    let radicand_value = numeric_limit_value(ctx, radicand)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let half_exponent = exponent.unsigned_abs() / 2;
    if half_exponent > FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
        return None;
    }

    let mut value = rational_pow_nonnegative(&radicand_value, half_exponent);
    if exponent < 0 {
        if value.is_zero() {
            return None;
        }
        value = BigRational::one() / value;
    }

    Some(ctx.add(Expr::Number(value)))
}

fn finite_cbrt_multiple_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if exponent % 3 != 0 {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base_limit).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Cbrt) || args.len() != 1 {
        return None;
    }

    let radicand_value = numeric_limit_value(ctx, args[0])?;
    if exponent <= 0 && radicand_value.is_zero() {
        return None;
    }

    let reduced_exponent = exponent.unsigned_abs() / 3;
    if reduced_exponent > FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
        return None;
    }

    let mut value = rational_pow_nonnegative(&radicand_value, reduced_exponent);
    if exponent < 0 {
        if value.is_zero() {
            return None;
        }
        value = BigRational::one() / value;
    }

    Some(ctx.add(Expr::Number(value)))
}

fn finite_integer_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if let Some(result) = finite_cbrt_multiple_power_result(ctx, base_limit, exponent) {
        return Some(result);
    }

    let base_nonzero = finite_denominator_proven_nonzero(ctx, base_limit);
    if exponent <= 0 && !base_nonzero {
        return None;
    }

    if exponent == 0 {
        return Some(ctx.num(1));
    }

    if let Some(result) = finite_sqrt_even_power_result(ctx, base_limit, exponent) {
        return Some(result);
    }

    if let Some(base_value) = numeric_limit_value(ctx, base_limit) {
        let abs_exponent = exponent.unsigned_abs();
        if abs_exponent <= FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
            let mut value = rational_pow_nonnegative(&base_value, abs_exponent);
            if exponent < 0 {
                if value.is_zero() {
                    return None;
                }
                value = BigRational::one() / value;
            }
            return Some(ctx.add(Expr::Number(value)));
        }
    }

    let exponent_expr = if exponent > 0 {
        ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
            exponent,
        ))))
    } else {
        let positive_exponent = exponent.checked_neg()?;
        let positive_exponent_expr = ctx.add(Expr::Number(BigRational::from_integer(
            BigInt::from(positive_exponent),
        )));
        let denominator = if positive_exponent == 1 {
            base_limit
        } else {
            ctx.add(Expr::Pow(base_limit, positive_exponent_expr))
        };
        let one = ctx.num(1);
        return Some(ctx.add(Expr::Div(one, denominator)));
    };

    Some(ctx.add(Expr::Pow(base_limit, exponent_expr)))
}

pub(super) fn apply_finite_integer_power_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let (base_expr, exponent) = parse_pow_int(ctx, expr)?;
    let base_limit = try_limit_rules_at_finite(ctx, base_expr, var, point)?;
    finite_integer_power_result(ctx, base_limit, exponent)
}

pub(super) fn apply_finite_elementary_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let (builtin, argument_expr) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            (ctx.builtin_of(fn_id)?, args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            (BuiltinFn::Exp, exp)
        }
        _ => return None,
    };
    if !matches!(
        builtin,
        BuiltinFn::Exp
            | BuiltinFn::Sin
            | BuiltinFn::Cos
            | BuiltinFn::Sinh
            | BuiltinFn::Cosh
            | BuiltinFn::Tanh
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Asinh
            | BuiltinFn::Cbrt
            | BuiltinFn::Abs
            | BuiltinFn::Ln
            | BuiltinFn::Log2
            | BuiltinFn::Log10
            | BuiltinFn::Sqrt
            | BuiltinFn::Asin
            | BuiltinFn::Arcsin
            | BuiltinFn::Acos
            | BuiltinFn::Arccos
            | BuiltinFn::Atanh
            | BuiltinFn::Acosh
            | BuiltinFn::Tan
            | BuiltinFn::Sec
    ) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, argument_expr, var_name).ok()?;
    let argument_value = argument.eval(&point_value);
    if is_finite_total_real_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return Some(finite_total_real_unary_result(ctx, builtin, argument_limit));
    }
    if is_finite_positive_domain_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return finite_positive_domain_unary_result(ctx, builtin, argument_limit);
    }
    if is_finite_partial_domain_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return finite_partial_domain_unary_result(ctx, builtin, argument_limit);
    }
    if is_finite_domain_checked_trig_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return finite_domain_checked_trig_unary_result(ctx, builtin, argument_limit);
    }

    None
}

pub(super) fn apply_finite_cube_root_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let argument_expr = pow_one_third_argument(ctx, expr)?;
    let argument_limit = try_limit_rules_at_finite(ctx, argument_expr, var, point)?;
    Some(finite_total_real_unary_result(
        ctx,
        BuiltinFn::Cbrt,
        argument_limit,
    ))
}

pub(super) fn apply_finite_square_root_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if !matches!(ctx.get(expr), Expr::Pow(_, _)) {
        return None;
    }

    let argument_expr = extract_square_root_base(ctx, expr)?;
    let argument_limit = try_limit_rules_at_finite(ctx, argument_expr, var, point)?;
    finite_positive_domain_unary_result(ctx, BuiltinFn::Sqrt, argument_limit)
}

/// All multi-indices of dimension `d` with total degree EXACTLY `k`, in a
/// deterministic (lexicographic) order.
pub(super) fn multi_indices_of_degree(d: usize, k: u32) -> Vec<Vec<u32>> {
    if d == 1 {
        return vec![vec![k]];
    }
    let mut out = Vec::new();
    for first in (0..=k).rev() {
        for mut rest in multi_indices_of_degree(d - 1, k - first) {
            let mut alpha = Vec::with_capacity(d);
            alpha.push(first);
            alpha.append(&mut rest);
            out.push(alpha);
        }
    }
    out
}

/// Divide two power series `num / den` (den_0 != 0) to `order` terms.
pub(super) fn power_series_divide(
    num: &[BigRational],
    den: &[BigRational],
    order: usize,
) -> Option<Vec<BigRational>> {
    use num_traits::Zero;
    let den0 = den.first()?;
    if den0.is_zero() {
        return None;
    }
    let mut q = vec![BigRational::zero(); order + 1];
    for n in 0..=order {
        let mut acc = num.get(n).cloned().unwrap_or_else(BigRational::zero);
        for (k, item) in den.iter().enumerate().take(n + 1).skip(1) {
            acc -= item * &q[n - k];
        }
        q[n] = acc / den0;
    }
    Some(q)
}

/// Keep only the coefficients up to `order` (drop higher-degree terms).
pub(super) fn truncate_polynomial(poly: &Polynomial, order: usize, var_name: &str) -> Polynomial {
    let coeffs: Vec<BigRational> = poly.coeffs.iter().take(order + 1).cloned().collect();
    Polynomial::new(coeffs, var_name.to_string())
}

/// The exponent of a `(var-point)` power factor: the bare shift (1), its
/// sqrt (1/2), or `(var-point)^p` for rational p. None when the factor is
/// not such a power.
pub(super) fn shift_power_exponent(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<BigRational> {
    if is_var_shift(ctx, expr, var, point) {
        return Some(rational_one());
    }
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_var_shift(ctx, *base, var, point) => {
            crate::numeric_eval::as_rational_const(ctx, *exp)
        }
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sqrt))
                && is_var_shift(ctx, args[0], var, point) =>
        {
            Some(BigRational::new(1.into(), 2.into()))
        }
        _ => None,
    }
}

/// Recognize (var - point)^p with rational p (var itself when point = 0),
/// including the sqrt form; returns p.
pub(super) fn one_sided_positive_power_of_shift(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<BigRational> {
    if is_var_shift(ctx, expr, var, point) {
        return Some(rational_one());
    }
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) if is_var_shift(ctx, base, var, point) => {
            crate::numeric_eval::as_rational_const(ctx, exp)
        }
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt))
                && is_var_shift(ctx, args[0], var, point) =>
        {
            Some(BigRational::new(1.into(), 2.into()))
        }
        _ => None,
    }
}

/// Rule 3: Power - lim x^n for integer n.
///
/// - n > 0: ±∞ (sign depends on approach and parity)
/// - n = 0: 1
/// - n < 0: 0
pub(crate) fn apply_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let (base, n) = parse_pow_int(ctx, expr)?;

    // Base must be exactly the limit variable
    if base != var {
        return None;
    }

    if n == 0 {
        return Some(ctx.num(1));
    }
    if n < 0 {
        return Some(ctx.num(0));
    }

    let sign = limit_sign(approach, n);
    Some(mk_infinity(ctx, sign))
}

pub(super) fn polynomial_growth_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
) -> Option<PolynomialGrowthInfo> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    let poly = multipoly_from_expr(ctx, expr, &budget).ok()?;
    if poly.is_zero() {
        return None;
    }

    let var_idx = poly.var_index(var_name)?;
    let degree = poly.degree_in(var_idx);
    if degree == 0 {
        return None;
    }

    let leading_coeff = poly.leading_coeff_in(var_idx).constant_value()?;
    Some(PolynomialGrowthInfo {
        degree,
        leading_coeff,
    })
}

fn negate_polynomial_growth_info(mut growth: PolynomialGrowthInfo) -> PolynomialGrowthInfo {
    growth.leading_coeff = -growth.leading_coeff;
    growth
}

fn scale_polynomial_growth_info(
    mut growth: PolynomialGrowthInfo,
    scale: BigRational,
) -> Option<PolynomialGrowthInfo> {
    if scale.is_zero() {
        return None;
    }
    growth.leading_coeff *= scale;
    Some(growth)
}

pub(super) fn polynomial_growth_info_with_bounded_additive_noise(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<PolynomialGrowthInfo> {
    if let Some(growth) = polynomial_growth_info(ctx, expr, var) {
        return Some(growth);
    }

    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(growth) =
                polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var, approach)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, rhs, var, approach) {
                    return Some(growth);
                }
            }
            if let Some(growth) =
                polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var, approach)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, lhs, var, approach) {
                    return Some(growth);
                }
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(growth) =
                polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var, approach)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, rhs, var, approach) {
                    return Some(growth);
                }
            }
            if let Some(growth) =
                polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var, approach)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, lhs, var, approach) {
                    return Some(negate_polynomial_growth_info(growth));
                }
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                return scale_polynomial_growth_info(
                    polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var, approach)?,
                    scale,
                );
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                return scale_polynomial_growth_info(
                    polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var, approach)?,
                    scale,
                );
            }
            None
        }
        Expr::Neg(inner) => Some(negate_polynomial_growth_info(
            polynomial_growth_info_with_bounded_additive_noise(ctx, inner, var, approach)?,
        )),
        _ => None,
    }
}

/// One orientation: `sqrt_side - linear_side` when `sqrt_first`, else
/// `linear_side - sqrt_side`. Returns the finite limit or None.
pub(super) fn sqrt_quadratic_minus_linear_oriented(
    ctx: &mut Context,
    sqrt_side: ExprId,
    linear_side: ExprId,
    sqrt_first: bool,
    var_name: &str,
    approach: InfSign,
) -> Option<ExprId> {
    let (sqrt_scale, radicand) = scaled_square_root_base(ctx, sqrt_side)?;
    if !sqrt_scale.is_positive() {
        return None;
    }
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    if radicand_poly.degree() != 2 {
        return None;
    }
    let a = radicand_poly.coeffs.get(2)?.clone();
    if !a.is_positive() {
        return None;
    }
    let b = radicand_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(|| BigRational::from_integer(BigInt::from(0)));
    let sqrt_a = rational_sqrt(&a)?;

    let linear_poly = Polynomial::from_expr(ctx, linear_side, var_name).ok()?;
    if linear_poly.degree() > 1 {
        return None;
    }
    let zero = BigRational::from_integer(BigInt::from(0));
    let d = linear_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(|| zero.clone());
    let e = linear_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(|| zero.clone());

    let two = BigRational::from_integer(BigInt::from(2));
    // sqrt(a x^2 + b x + c) ~ sqrt(a)|x| + sign(x) b/(2 sqrt(a)).
    let (sqrt_leading, b_constant) = match approach {
        InfSign::Pos => (&sqrt_scale * &sqrt_a, &sqrt_scale * &b / (&two * &sqrt_a)),
        InfSign::Neg => (
            -(&sqrt_scale * &sqrt_a),
            -(&sqrt_scale * &b) / (&two * &sqrt_a),
        ),
    };
    let (leading, constant) = if sqrt_first {
        (&sqrt_leading - &d, &b_constant - &e)
    } else {
        (&d - &sqrt_leading, &e - &b_constant)
    };
    // Finite limit only when the leading terms cancel exactly.
    if !leading.is_zero() {
        return None;
    }
    Some(ctx.add(Expr::Number(constant)))
}

pub(super) fn linear_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let growth = polynomial_growth_info(ctx, arg, var)?;
    if growth.degree != 1 {
        return None;
    }
    Some(limit_growth_sign(
        &growth.leading_coeff,
        growth.degree,
        approach,
    ))
}

pub(super) fn polynomial_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let growth = polynomial_growth_info(ctx, arg, var)?;
    Some(limit_growth_sign(
        &growth.leading_coeff,
        growth.degree,
        approach,
    ))
}

pub(super) fn polynomial_or_constant_growth_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
) -> Option<PolynomialGrowthInfo> {
    polynomial_growth_info(ctx, expr, var).or_else(|| {
        let leading_coeff = constant_rational_value(ctx, expr)?;
        if leading_coeff.is_zero() {
            return None;
        }
        Some(PolynomialGrowthInfo {
            degree: 0,
            leading_coeff,
        })
    })
}

pub(super) fn polynomial_or_numeric_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    if let Some(growth) = polynomial_growth_info(ctx, expr, var) {
        return Some(limit_growth_sign(
            &growth.leading_coeff,
            growth.degree,
            approach,
        ));
    }

    let value = numeric_limit_value(ctx, expr)?;
    if value.is_zero() {
        None
    } else if value.is_positive() {
        Some(InfSign::Pos)
    } else {
        Some(InfSign::Neg)
    }
}

fn subpolynomial_tail_coeff(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<BigRational> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            let builtin = ctx.builtin_of(fn_id)?;
            match (builtin, args.as_slice()) {
                (BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10, [arg]) => {
                    log_argument_tail_coeff(ctx, *arg, var, approach)
                }
                (BuiltinFn::Acosh, [arg])
                    if unbounded_argument_tail_sign(ctx, *arg, var, approach)? == InfSign::Pos =>
                {
                    Some(rational_one())
                }
                (BuiltinFn::Sqrt, [arg])
                    if linear_argument_tail_sign(ctx, *arg, var, approach)? == InfSign::Pos =>
                {
                    Some(rational_one())
                }
                (BuiltinFn::Cbrt | BuiltinFn::Asinh, [arg]) => {
                    match linear_argument_tail_sign(ctx, *arg, var, approach)? {
                        InfSign::Pos => Some(rational_one()),
                        InfSign::Neg => Some(-rational_one()),
                    }
                }
                (BuiltinFn::Log, [base, arg]) => {
                    let arg_coeff = log_argument_tail_coeff(ctx, *arg, var, approach)?;
                    Some(positive_log_base_tail_coeff(ctx, *base)? * arg_coeff)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

pub(super) fn scaled_subpolynomial_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledSubpolynomialTailInfo> {
    if let Some(coeff) = subpolynomial_tail_coeff(ctx, expr, var, approach) {
        return Some(ScaledSubpolynomialTailInfo { coeff });
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut info = scaled_subpolynomial_tail_info(ctx, inner, var, approach)?;
            info.coeff = -info.coeff;
            Some(info)
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(lhs_scale) = numeric_limit_value(ctx, lhs) {
                if let Some(mut rhs_info) = scaled_subpolynomial_tail_info(ctx, rhs, var, approach)
                {
                    rhs_info.coeff *= lhs_scale;
                    return Some(rhs_info);
                }
            }
            if let Some(rhs_scale) = numeric_limit_value(ctx, rhs) {
                if let Some(mut lhs_info) = scaled_subpolynomial_tail_info(ctx, lhs, var, approach)
                {
                    lhs_info.coeff *= rhs_scale;
                    return Some(lhs_info);
                }
            }
            None
        }
        _ => None,
    }
}

pub(super) fn nonzero_scaled_subpolynomial_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledSubpolynomialTailInfo> {
    let info = scaled_subpolynomial_tail_info(ctx, expr, var, approach)?;
    if info.coeff.is_zero() {
        None
    } else {
        Some(info)
    }
}

pub(super) fn subpolynomial_tail_sign(info: &ScaledSubpolynomialTailInfo) -> Option<InfSign> {
    if info.coeff.is_zero() {
        None
    } else if info.coeff.is_positive() {
        Some(InfSign::Pos)
    } else {
        Some(InfSign::Neg)
    }
}

/// `c * x^b` with `b > 0` rational (fractional included): the bare variable,
/// its rational powers, `sqrt(var)`, and a numeric scale. Returns (c, b).
pub(super) fn positive_power_tail(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
) -> Option<(BigRational, BigRational)> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (c, b) = positive_power_tail(ctx, inner, var)?;
            Some((-c, b))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                let (c, b) = positive_power_tail(ctx, rhs, var)?;
                return Some((scale * c, b));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                let (c, b) = positive_power_tail(ctx, lhs, var)?;
                return Some((scale * c, b));
            }
            None
        }
        Expr::Pow(base, exponent) if base == var => {
            let exp = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            exp.is_positive().then_some((rational_one(), exp))
        }
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sqrt))
                && args[0] == var =>
        {
            Some((
                rational_one(),
                BigRational::new(BigInt::from(1), BigInt::from(2)),
            ))
        }
        _ if expr == var => Some((rational_one(), rational_one())),
        _ => None,
    }
}

/// The sign of a positive-degree polynomial as `x -> +inf` (the sign of its
/// leading coefficient, since `x^deg -> +inf`). None for a non-polynomial or a
/// constant.
pub(super) fn positive_degree_polynomial_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<i32> {
    use num_traits::Signed;
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() < 1 {
        return None;
    }
    let leading = poly.coeffs.get(poly.degree())?;
    Some(if leading.is_positive() { 1 } else { -1 })
}

/// The `0^0` form `x^g -> exp(lim g * ln x)` as `x -> 0+`. The base is the bare
/// variable, which is positive on the RIGHT of 0, so `x^g = exp(g ln x)` is
/// real and the limit is `exp(lim g ln x)`. Resolves the canonical `x^x -> 1`
/// (lim x ln x = 0). Fires only on the right side at 0 with a bare-variable
/// base; a two-sided `x^x` is complex on the left and stays residual.
pub(super) fn apply_finite_zero_base_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    use num_traits::{One, Zero};
    if !matches!(side, FiniteLimitSide::Right) {
        return None;
    }
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    // The base is the bare variable: x -> 0+ through positive values, so ln(x)
    // is real. A non-variable base could approach 0 with an unknown sign.
    if base != var || !depends_on(ctx, exp, var) {
        return None;
    }
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![var]);
    let product = ctx.add(Expr::Mul(exp, ln_base));
    let l_limit = try_limit_rules_at_finite_one_sided(ctx, product, var, point, side)?;
    match limit_value_infinite_sign(ctx, l_limit) {
        Some(1) => Some(mk_infinity(ctx, InfSign::Pos)),
        Some(_) => Some(ctx.num(0)),
        None => {
            if let Some(value) = crate::numeric_eval::as_rational_const(ctx, l_limit) {
                if value.is_zero() {
                    return Some(ctx.num(1));
                }
                if value.is_one() {
                    return Some(ctx.add(Expr::Constant(Constant::E)));
                }
            }
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, l_limit)))
        }
    }
}

/// Entire sub-shape: polynomial tree (no `Div`, no calls).
pub(super) fn expr_is_entire_polynomial_shape(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::Signed;
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) => true,
        Expr::Constant(c) => !matches!(c, Constant::Infinity | Constant::Undefined),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            expr_is_entire_polynomial_shape(ctx, *l) && expr_is_entire_polynomial_shape(ctx, *r)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_is_entire_polynomial_shape(ctx, *inner),
        Expr::Pow(base, exp) => {
            crate::numeric_eval::as_rational_const(ctx, *exp)
                .is_some_and(|e| e.is_integer() && !e.is_negative())
                && expr_is_entire_polynomial_shape(ctx, *base)
        }
        _ => false,
    }
}

pub(super) fn finite_polynomial_tail_negative_on_both_sides(
    polynomial: &Polynomial,
    point_value: &BigRational,
) -> Option<bool> {
    let (order, derivative) =
        finite_polynomial_local_order_and_derivative(polynomial, point_value)?;
    finite_local_tail_negative_on_both_sides(&derivative, order)
}

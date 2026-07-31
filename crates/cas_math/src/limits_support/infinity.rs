//! `limits_support`: familia `infinity`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

/// 0/0 finite-point limits resolved by replacing every factor of the numerator
/// and denominator with its first-order equivalent infinitesimal, then taking
/// the polynomial ratio. Generalizes the sine/exp small-angle rules to handle
/// inversion (`x/sin(x) -> 1`), composition (`sin(3x)/sin(5x) -> 3/5`), and the
/// missing equivalents (`tan/asin/arctan/sinh/tanh`).
///
/// Runs AFTER the sine/exp/log rules, so it only fires on the cases they leave
/// residual (a non-polynomial denominator, or a numerator with no existing
/// recognizer). Only acts on a genuine 0/0 form: both equivalents must vanish
/// at the point.
pub(super) fn apply_finite_equivalent_infinitesimal_quotient_rule(
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
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let numerator = first_order_equivalent_poly(ctx, num, &var_name, &point_value)?;
    let denominator = first_order_equivalent_poly(ctx, den, &var_name, &point_value)?;
    // Genuine 0/0 only: a non-vanishing denominator is a continuous case owned
    // by ordinary substitution, and a non-vanishing numerator over a vanishing
    // denominator is a pole that must stay residual.
    if !numerator.eval(&point_value).is_zero() || !denominator.eval(&point_value).is_zero() {
        return None;
    }
    let value = finite_rational_polynomial_value(&numerator, &denominator, &point_value)?;
    Some(ctx.add(Expr::Number(value)))
}

/// One-sided composition: scaling by constants, additive combination/// One-sided composition: scaling by constants, additive combination
/// with infinity awareness (infinity - infinity refuses), and the
/// power-log dominance u^p * ln(u)^q -> 0 (p > 0) as u -> 0+ with
/// u = var - point. Children resolve recursively through the full
/// one-sided chain, so the existing endpoint atoms compose.
/// The signed infinity of the one-sided limit of `inner`, or None when
/// it is finite or undecidable.
pub(super) fn one_sided_inner_infinity_sign(
    ctx: &mut Context,
    inner: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    let value = try_limit_rules_at_finite_one_sided(ctx, inner, var, point, side)?;
    infinity_sign_of_expr(ctx, value)
}

/// Build `outer(+-inf)` and resolve it via the saturation fold; returns
/// the folded value only when the fold actually changed it (so
/// oscillating outers like sin/cos, which do not fold, decline).
pub(super) fn saturate_outer_at_infinity(
    ctx: &mut Context,
    build_outer: impl FnOnce(&mut Context, ExprId) -> ExprId,
    sign: InfSign,
) -> Option<ExprId> {
    let inf = mk_infinity(ctx, sign);
    let candidate = build_outer(ctx, inf);
    let folded = crate::infinity_support::fold_infinity_saturation(ctx, candidate);
    (folded != candidate).then_some(folded)
}

pub(super) fn infinity_sign_of_expr(ctx: &Context, expr: ExprId) -> Option<InfSign> {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity) => Some(InfSign::Pos),
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            Some(InfSign::Neg)
        }
        _ => None,
    }
}

pub(super) fn scale_infinity(
    ctx: &mut Context,
    scale: &BigRational,
    sign: InfSign,
) -> Option<ExprId> {
    if scale.is_zero() {
        return None;
    }

    let result_sign = if scale.is_positive() {
        sign
    } else {
        neg_inf_sign(sign)
    };
    Some(mk_infinity(ctx, result_sign))
}

/// Limit of `sqrt(P) - sqrt(Q)` at +-infinity for polynomials P, Q with
/// the SAME positive leading coefficient and the same degree n in
/// {1, 2}. The conjugate expansion gives sqrt(P) = sqrt(a) x^(n/2) +
/// (b_P/(2 sqrt(a))) x^(n/2 - 1) + ..., so the difference is finite when
/// n <= 2: degree-1 radicands always cancel to 0, degree-2 radicands
/// give (b_P - b_Q)/(2 sqrt(a)). Covers sqrt(x+1) - sqrt(x) = 0 and
/// sqrt(x^2+x) - sqrt(x^2-x) = 1. Higher degrees and mismatched leading
/// terms decline.
pub(super) fn sqrt_minus_sqrt_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let (left, right) = match ctx.get(expr).clone() {
        Expr::Sub(l, r) => (l, r),
        Expr::Add(l, r) => match ctx.get(r).clone() {
            Expr::Neg(inner) => (l, inner),
            _ => match ctx.get(l).clone() {
                Expr::Neg(inner) => (inner, r),
                _ => return None,
            },
        },
        _ => return None,
    };
    let one = BigRational::from_integer(BigInt::from(1));
    let (left_scale, left_radicand) = scaled_square_root_base(ctx, left)?;
    let (right_scale, right_radicand) = scaled_square_root_base(ctx, right)?;
    if left_scale != one || right_scale != one {
        return None;
    }
    let p = Polynomial::from_expr(ctx, left_radicand, &var_name).ok()?;
    let q = Polynomial::from_expr(ctx, right_radicand, &var_name).ok()?;
    let degree = p.degree();
    if degree != q.degree() || degree == 0 || degree > 2 {
        return None;
    }
    let a = p.coeffs.get(degree)?.clone();
    if !a.is_positive() || q.coeffs.get(degree) != Some(&a) {
        return None;
    }
    let zero = BigRational::from_integer(BigInt::from(0));
    if degree == 1 {
        // sqrt(linear) - sqrt(linear): both -> +inf, difference -> 0.
        // Only defined as x -> +inf (radicand goes negative at -inf).
        if approach != InfSign::Pos {
            return None;
        }
        return Some(ctx.add(Expr::Number(zero)));
    }
    // degree == 2: (b_P - b_Q)/(2 sqrt(a)), sign flips at -inf.
    let sqrt_a = rational_sqrt(&a)?;
    let b_p = p.coeffs.get(1).cloned().unwrap_or_else(|| zero.clone());
    let b_q = q.coeffs.get(1).cloned().unwrap_or_else(|| zero.clone());
    let two = BigRational::from_integer(BigInt::from(2));
    let mut value = (&b_p - &b_q) / (&two * &sqrt_a);
    if approach == InfSign::Neg {
        value = -value;
    }
    Some(ctx.add(Expr::Number(value)))
}

/// Limit of `sqrt(a x^2 + b x + c) - (d x + e)` (or the reverse) at
/// +-infinity via the conjugate / leading-term expansion. With sqrt(a)
/// rational and the leading terms cancelling, the limit is finite:
/// sqrt(a x^2 + b x + c) = sqrt(a)|x| + b/(2 sqrt(a)) sign(x) + o(1).
/// Covers the classic sqrt(x^2+x) - x = 1/2, sqrt(x^2+1) - x = 0,
/// x - sqrt(x^2-x) = 1/2. Diverging cases (leading terms differ) decline
/// here and fall through to the polynomial-dominance rules.
pub(super) fn sqrt_quadratic_minus_linear_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let (left, right) = match ctx.get(expr).clone() {
        Expr::Sub(l, r) => (l, r),
        Expr::Add(l, r) => match ctx.get(r).clone() {
            Expr::Neg(inner) => (l, inner),
            _ => match ctx.get(l).clone() {
                Expr::Neg(inner) => (r, inner),
                // Pure `sqrt(...) + linear` (both positive). At x → −∞ the
                // square root behaves like −x (|x| = −x), so the leading
                // terms can cancel and the sum is finite —
                // `sqrt(x²+1) + x → 0`, `sqrt(x²+x) + x → −1/2`. Treat it as
                // the difference `sqrt − (−linear)` by negating the linear
                // side; the leading-cancellation guard inside `_oriented`
                // keeps the genuinely divergent cases (e.g. this shape at
                // +∞, or `sqrt(x²+1) + 2x` at −∞) honestly residual.
                _ => {
                    let neg_r = ctx.add(Expr::Neg(r));
                    let neg_l = ctx.add(Expr::Neg(l));
                    return sqrt_quadratic_minus_linear_oriented(
                        ctx, l, neg_r, true, &var_name, approach,
                    )
                    .or_else(|| {
                        sqrt_quadratic_minus_linear_oriented(
                            ctx, r, neg_l, true, &var_name, approach,
                        )
                    });
                }
            },
        },
        _ => return None,
    };
    sqrt_quadratic_minus_linear_oriented(ctx, left, right, true, &var_name, approach).or_else(
        || sqrt_quadratic_minus_linear_oriented(ctx, right, left, false, &var_name, approach),
    )
}

/// Limit of `factor * (radical difference)` at +infinity when the difference is
/// a conjugate form that DECAYS to 0 — the genuine `0 * inf` form the
/// multiplicative rule leaves residual (e.g. `x (sqrt(x^2+1) - x)`). Rationalizing
/// the difference by its conjugate turns it into `N(x) / (sqrt + ...)`, whose
/// leading asymptotic `K x^p` multiplies the factor's leading `c x^q`; the limit
/// is the coefficient `c K` when `p + q == 0`, is `0` when `p + q < 0`, and the
/// divergent case `p + q > 0` declines to the dominance rules. Covers the classic
/// `x (sqrt(x^2+1) - x) = 1/2`, `x (sqrt(x^2+a) - x) = a/2`,
/// `x (sqrt(x^2+2x) - x - 1) = -1/2`, and `sqrt(x) (sqrt(x+1) - sqrt(x)) = 1/2`.
/// Only defined as `x -> +inf` (radicands and sqrt factors require `x > 0`);
/// the `-inf` side declines and stays honest.
pub(super) fn radical_conjugate_product_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let (a, b) = match ctx.get(expr).clone() {
        Expr::Mul(a, b) => (a, b),
        _ => return None,
    };
    radical_conjugate_product_oriented(ctx, a, b, &var_name)
        .or_else(|| radical_conjugate_product_oriented(ctx, b, a, &var_name))
}

/// Limit at +infinity of `[factor *] (cube-root conjugate difference)` -- the
/// cube-root companion of `radical_conjugate_product_limit_at_infinity`. Handles
/// the bare difference (`cbrt(x^3+x^2) - x = 1/3`) and the `0 * inf` product
/// (`x^2 (cbrt(x^3+1) - x) = 1/3`). Only fires toward +inf and declines the
/// divergent (exponent-sum > 0) case to the dominance rules.
pub(super) fn cbrt_conjugate_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let zero = BigRational::from_integer(BigInt::from(0));

    // Bare difference (no growing factor).
    if let Some((coeff, exp)) = cbrt_difference_asymptotic_at_pos_inf(ctx, expr, &var_name) {
        if exp > zero {
            return None;
        }
        if exp < zero {
            return Some(ctx.add(Expr::Number(zero)));
        }
        return Some(ctx.add(Expr::Number(coeff)));
    }

    // factor * difference, in either order.
    if let Expr::Mul(a, b) = ctx.get(expr).clone() {
        return cbrt_conjugate_product_oriented(ctx, a, b, &var_name)
            .or_else(|| cbrt_conjugate_product_oriented(ctx, b, a, &var_name));
    }
    None
}

/// Limit at +infinity of `[factor *] ((P)^(1/n) - L)` for a general integer
/// `n >= 2`, the Pow-form companion of the sqrt and cbrt conjugate rules. Runs
/// after them, so it only newly resolves the higher roots they do not own.
pub(super) fn nth_root_conjugate_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym).to_string();
    let zero = BigRational::from_integer(BigInt::from(0));

    if let Some((coeff, exp)) = nth_root_difference_asymptotic_at_pos_inf(ctx, expr, &var_name) {
        if exp > zero {
            return None;
        }
        if exp < zero {
            return Some(ctx.add(Expr::Number(zero)));
        }
        return Some(ctx.add(Expr::Number(coeff)));
    }

    if let Expr::Mul(a, b) = ctx.get(expr).clone() {
        return nth_root_conjugate_product_oriented(ctx, a, b, &var_name)
            .or_else(|| nth_root_conjugate_product_oriented(ctx, b, a, &var_name));
    }
    None
}

pub(super) fn sqrt_polynomial_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (sqrt_scale, radicand) = scaled_square_root_base(ctx, num)?;
    let radicand_growth =
        polynomial_growth_info_with_bounded_additive_noise(ctx, radicand, var, approach)?;
    let den_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, den, var, approach)?;

    if radicand_growth.degree == 0 || radicand_growth.degree % 2 != 0 {
        return None;
    }
    if !radicand_growth.leading_coeff.is_positive() || den_growth.leading_coeff.is_zero() {
        return None;
    }

    let sqrt_degree = radicand_growth.degree / 2;
    if den_growth.degree != sqrt_degree {
        return None;
    }

    if let Some(sqrt_leading_coeff) = rational_sqrt(&radicand_growth.leading_coeff) {
        let mut ratio = sqrt_scale * sqrt_leading_coeff / den_growth.leading_coeff;
        if approach == InfSign::Neg && sqrt_degree % 2 == 1 {
            ratio = -ratio;
        }
        return Some(ctx.add(Expr::Number(ratio)));
    }

    let leading_coeff = ctx.add(Expr::Number(radicand_growth.leading_coeff));
    let sqrt_leading_coeff = ctx.call_builtin(BuiltinFn::Sqrt, vec![leading_coeff]);
    let denominator_abs = den_growth.leading_coeff.abs();
    let scale_abs = sqrt_scale.abs();
    let one = BigRational::from_integer(BigInt::from(1));
    let scaled_sqrt = if scale_abs == one {
        sqrt_leading_coeff
    } else {
        let multiplier = ctx.add(Expr::Number(scale_abs));
        ctx.add(Expr::Mul(multiplier, sqrt_leading_coeff))
    };
    let unsigned_result = if denominator_abs == one {
        scaled_sqrt
    } else {
        let denominator = ctx.add(Expr::Number(denominator_abs));
        ctx.add(Expr::Div(scaled_sqrt, denominator))
    };

    let flips_at_neg_infinity = approach == InfSign::Neg && sqrt_degree % 2 == 1;
    let needs_negation =
        sqrt_scale.is_negative() ^ den_growth.leading_coeff.is_negative() ^ flips_at_neg_infinity;
    if needs_negation {
        Some(ctx.add(Expr::Neg(unsigned_result)))
    } else {
        Some(unsigned_result)
    }
}

pub(super) fn polynomial_sqrt_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let radicand = extract_square_root_base(ctx, den)?;
    let num_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, num, var, approach)?;
    let radicand_growth =
        polynomial_growth_info_with_bounded_additive_noise(ctx, radicand, var, approach)?;

    if radicand_growth.degree == 0 || radicand_growth.degree % 2 != 0 {
        return None;
    }
    if !radicand_growth.leading_coeff.is_positive() || num_growth.leading_coeff.is_zero() {
        return None;
    }

    let sqrt_degree = radicand_growth.degree / 2;
    if num_growth.degree != sqrt_degree {
        return None;
    }

    let mut signed_num_lc = num_growth.leading_coeff;
    if approach == InfSign::Neg && sqrt_degree % 2 == 1 {
        signed_num_lc = -signed_num_lc;
    }

    if let Some(sqrt_leading_coeff) = rational_sqrt(&radicand_growth.leading_coeff) {
        return Some(ctx.add(Expr::Number(signed_num_lc / sqrt_leading_coeff)));
    }

    let coeff = signed_num_lc / radicand_growth.leading_coeff.clone();
    Some(rationalized_surd_product(
        ctx,
        coeff,
        radicand_growth.leading_coeff,
    ))
}

fn abs_polynomial_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (abs_scale, abs_arg) = scaled_abs_base(ctx, num)?;
    let abs_arg_growth =
        polynomial_growth_info_with_bounded_additive_noise(ctx, abs_arg, var, approach)?;
    let den_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, den, var, approach)?;

    let numerator_tail = if abs_scale.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    let denominator_tail =
        limit_growth_sign(&den_growth.leading_coeff, den_growth.degree, approach);

    if abs_arg_growth.degree < den_growth.degree {
        return Some(ctx.num(0));
    }
    if abs_arg_growth.degree > den_growth.degree {
        return Some(signed_abs_ratio_infinity(
            ctx,
            numerator_tail,
            denominator_tail,
        ));
    }

    let magnitude =
        abs_scale.abs() * abs_arg_growth.leading_coeff.abs() / den_growth.leading_coeff.abs();
    Some(signed_abs_ratio_result(
        ctx,
        magnitude,
        numerator_tail,
        denominator_tail,
    ))
}

fn polynomial_abs_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (abs_scale, abs_arg) = scaled_abs_base(ctx, den)?;
    let num_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, num, var, approach)?;
    let abs_arg_growth =
        polynomial_growth_info_with_bounded_additive_noise(ctx, abs_arg, var, approach)?;

    let numerator_tail = limit_growth_sign(&num_growth.leading_coeff, num_growth.degree, approach);
    let denominator_tail = if abs_scale.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };

    if num_growth.degree < abs_arg_growth.degree {
        return Some(ctx.num(0));
    }
    if num_growth.degree > abs_arg_growth.degree {
        return Some(signed_abs_ratio_infinity(
            ctx,
            numerator_tail,
            denominator_tail,
        ));
    }

    let magnitude =
        num_growth.leading_coeff.abs() / (abs_scale.abs() * abs_arg_growth.leading_coeff.abs());
    Some(signed_abs_ratio_result(
        ctx,
        magnitude,
        numerator_tail,
        denominator_tail,
    ))
}

pub(super) fn unbounded_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    polynomial_argument_tail_sign(ctx, arg, var, approach)
        .or_else(|| rational_polynomial_argument_tail_sign(ctx, arg, var, approach))
        .or_else(|| abs_unbounded_argument_tail_sign(ctx, arg, var, approach))
        .or_else(|| radical_unbounded_argument_tail_sign(ctx, arg, var, approach))
}

/// sqrt(inner) and inner^q (rational q > 0) tend to +infinity whenever
/// the inner argument is unbounded toward +infinity, which lets
/// saturating compositions like arctan(sqrt(x)) resolve.
fn radical_unbounded_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(arg) {
        Expr::Neg(inner) => match unbounded_argument_tail_sign(ctx, *inner, var, approach)? {
            InfSign::Pos => Some(InfSign::Neg),
            InfSign::Neg => Some(InfSign::Pos),
        },
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sqrt)) =>
        {
            (unbounded_argument_tail_sign(ctx, args[0], var, approach)? == InfSign::Pos)
                .then_some(InfSign::Pos)
        }
        Expr::Pow(base, exponent) => {
            let value = crate::numeric_eval::as_rational_const(ctx, *exponent)?;
            if !value.is_positive() || value.is_integer() {
                return None;
            }
            (unbounded_argument_tail_sign(ctx, *base, var, approach)? == InfSign::Pos)
                .then_some(InfSign::Pos)
        }
        _ => None,
    }
}

/// |inner| tends to +infinity whenever the inner argument is unbounded
/// toward either sign, which lets compositions like ln(|x|) resolve at
/// both infinities (antiderivatives emit ln|.| forms).
fn abs_unbounded_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(arg) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Abs)) =>
        {
            unbounded_argument_tail_sign(ctx, args[0], var, approach).map(|_| InfSign::Pos)
        }
        _ => None,
    }
}

fn unary_log_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if !matches!(builtin, BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) {
        return None;
    }

    let coeff = log_argument_tail_coeff(ctx, arg, var, approach)?;
    scale_infinity(ctx, &coeff, InfSign::Pos)
}

fn elementary_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if let Some(limit) = unary_log_argument_limit_at_infinity(ctx, builtin, arg, var, approach) {
        return Some(limit);
    }

    let arg_tail = match builtin {
        // Saturating maps (and exp's signed tails) only consume the
        // tail SIGN, so any certified unbounded inner is sound.
        BuiltinFn::Acosh
        | BuiltinFn::Abs
        | BuiltinFn::Exp
        | BuiltinFn::Atan
        | BuiltinFn::Arctan
        | BuiltinFn::Tanh => unbounded_argument_tail_sign(ctx, arg, var, approach)?,
        BuiltinFn::Cbrt | BuiltinFn::Asinh | BuiltinFn::Sinh | BuiltinFn::Cosh => {
            polynomial_argument_tail_sign(ctx, arg, var, approach)?
        }
        _ => linear_argument_tail_sign(ctx, arg, var, approach)?,
    };
    match (builtin, arg_tail) {
        (BuiltinFn::Sqrt | BuiltinFn::Acosh, InfSign::Pos) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Abs, _) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Atan | BuiltinFn::Arctan, tail) => Some(signed_pi_over_two(ctx, tail)),
        (BuiltinFn::Tanh, tail) => Some(signed_unit_limit(ctx, tail)),
        (BuiltinFn::Sinh, tail) => Some(mk_infinity(ctx, tail)),
        (BuiltinFn::Cosh, _) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Cbrt | BuiltinFn::Asinh, tail) => Some(mk_infinity(ctx, tail)),
        (BuiltinFn::Exp, InfSign::Pos) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Exp, InfSign::Neg) => Some(ctx.num(0)),
        _ => None,
    }
}

fn binary_log_argument_limit_at_infinity(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let arg_coeff = log_argument_tail_coeff(ctx, arg, var, approach)?;
    let base_coeff = positive_log_base_tail_coeff(ctx, base)?;
    scale_infinity(ctx, &(base_coeff * arg_coeff), InfSign::Pos)
}

fn unary_linear_exp_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if !matches!(
        builtin,
        BuiltinFn::Ln
            | BuiltinFn::Log2
            | BuiltinFn::Log10
            | BuiltinFn::Sqrt
            | BuiltinFn::Cbrt
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Tanh
            | BuiltinFn::Sinh
            | BuiltinFn::Cosh
            | BuiltinFn::Asinh
            | BuiltinFn::Acosh
    ) {
        return None;
    }

    let exp_tail = linear_exp_tail_sign(ctx, arg, var, approach)?;
    match builtin {
        BuiltinFn::Sqrt | BuiltinFn::Cbrt | BuiltinFn::Asinh if exp_tail == InfSign::Neg => {
            Some(ctx.num(0))
        }
        BuiltinFn::Atan | BuiltinFn::Arctan if exp_tail == InfSign::Neg => Some(ctx.num(0)),
        BuiltinFn::Atan | BuiltinFn::Arctan => Some(signed_pi_over_two(ctx, InfSign::Pos)),
        BuiltinFn::Tanh if exp_tail == InfSign::Neg => Some(ctx.num(0)),
        BuiltinFn::Tanh => Some(ctx.num(1)),
        BuiltinFn::Sinh if exp_tail == InfSign::Neg => Some(ctx.num(0)),
        BuiltinFn::Sinh => Some(mk_infinity(ctx, InfSign::Pos)),
        BuiltinFn::Cosh if exp_tail == InfSign::Neg => Some(ctx.num(1)),
        BuiltinFn::Cosh => Some(mk_infinity(ctx, InfSign::Pos)),
        BuiltinFn::Sqrt | BuiltinFn::Cbrt | BuiltinFn::Asinh => {
            Some(mk_infinity(ctx, InfSign::Pos))
        }
        BuiltinFn::Acosh if exp_tail == InfSign::Pos => Some(mk_infinity(ctx, InfSign::Pos)),
        BuiltinFn::Acosh => None,
        _ => Some(mk_infinity(ctx, exp_tail)),
    }
}

fn binary_log_linear_exp_argument_limit_at_infinity(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let exp_tail = linear_exp_tail_sign(ctx, arg, var, approach)?;
    let base_coeff = positive_log_base_tail_coeff(ctx, base)?;
    scale_infinity(ctx, &base_coeff, exp_tail)
}

fn unary_elementary_function_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if let Some(result) = elementary_argument_limit_at_infinity(ctx, builtin, arg, var, approach) {
        return Some(result);
    }

    unary_linear_exp_argument_limit_at_infinity(ctx, builtin, arg, var, approach)
}

fn unary_log_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if !matches!(builtin, BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) {
        return None;
    }

    // See through an `abs` wrapper: `lim ln(|u|) = ln(|lim u|)`. This is what a rational
    // antiderivative emits — `½·ln|(x-1)/(x+1)|` — so it must resolve at the infinite bound for
    // the improper integral `∫₂^∞ 1/(x²-1)` to evaluate.
    let (effective_arg, take_abs) = match ctx.get(arg).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(fn_id, BuiltinFn::Abs) => {
            (args[0], true)
        }
        _ => (arg, false),
    };

    let value = rational_polynomial_argument_finite_tail_value(ctx, effective_arg, var)?;
    let value = if take_abs {
        num_traits::Signed::abs(&value)
    } else {
        value
    };
    if !value.is_positive() {
        return None;
    }

    let value_expr = ctx.add(Expr::Number(value));
    finite_positive_domain_unary_result(ctx, builtin, value_expr)
}

fn unary_sqrt_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Sqrt {
        return None;
    }

    let value = rational_polynomial_argument_finite_tail_value(ctx, arg, var)?;
    if !value.is_positive() {
        return None;
    }

    let value_expr = ctx.add(Expr::Number(value));
    finite_positive_domain_unary_result(ctx, BuiltinFn::Sqrt, value_expr)
}

/// `lim_{x→∞} |u(x)| = |L|` when the rational argument `u` has a finite tail value
/// `L`. `|·|` is continuous, so the bare-infinity rule (`abs → +∞`, valid only
/// when the argument diverges) must NOT fire here. This is what closes
/// `lim ln(|(x-1)/(x+1)|) = ln(1) = 0` (the argument `(x-1)/(x+1) → 1`), and with
/// it the improper integral `∫₂^∞ 1/(x²-1) = ½·ln 3`, whose antiderivative
/// `½·ln|(x-1)/(x+1)|` is evaluated at the infinite bound through this limit.
fn unary_abs_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Abs {
        return None;
    }

    let value = rational_polynomial_argument_finite_tail_value(ctx, arg, var)?;
    Some(ctx.add(Expr::Number(num_traits::Signed::abs(&value))))
}

fn unary_sqrt_zero_tail_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Sqrt {
        return None;
    }

    if rational_polynomial_argument_zero_tail_sign(ctx, arg, var, approach)? != InfSign::Pos {
        return None;
    }

    Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))))
}

fn unary_cbrt_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Cbrt {
        return None;
    }

    if let Some(tail) = rational_polynomial_argument_tail_sign(ctx, arg, var, approach) {
        return Some(mk_infinity(ctx, tail));
    }

    if let Some(value) = rational_polynomial_argument_finite_tail_value(ctx, arg, var) {
        let value_expr = ctx.add(Expr::Number(value));
        return Some(finite_total_real_unary_result(
            ctx,
            BuiltinFn::Cbrt,
            value_expr,
        ));
    }

    rational_polynomial_argument_zero_tail_sign(ctx, arg, var, approach)?;
    Some(ctx.num(0))
}

fn unary_acosh_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if builtin != BuiltinFn::Acosh {
        return None;
    }

    let value = rational_polynomial_argument_finite_tail_value(ctx, arg, var)?;
    if value <= rational_one() {
        return None;
    }

    let value_expr = ctx.add(Expr::Number(value));
    finite_partial_domain_unary_result(ctx, BuiltinFn::Acosh, value_expr)
}

fn binary_log_finite_rational_argument_limit_at_infinity(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    var: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, base, var) {
        return None;
    }

    let value = rational_polynomial_argument_finite_tail_value(ctx, arg, var)?;
    if !value.is_positive() {
        return None;
    }

    let value_expr = ctx.add(Expr::Number(value));
    finite_log_result(ctx, base, value_expr)
}

/// Elementary argument limits as `x -> ±∞`.
///
/// Logs and inverse hyperbolic cosine accept polynomial and rational-polynomial
/// arguments only when leading-term analysis proves a positive unbounded real
/// tail. Logs also accept rational-polynomial arguments with positive finite or
/// zero tails through explicit degree/leading-coefficient analysis. Square
/// roots accept positive rational-polynomial finite tails and path-compatible
/// positive zero tails. Cube roots, inverse hyperbolic sine, arctangent,
/// hyperbolic tangent, hyperbolic sine, and hyperbolic cosine accept signed
/// polynomial tails because they are total over the reals; cube roots also
/// accept signed rational-polynomial unbounded, finite, and zero tails. Inverse hyperbolic
/// cosine also accepts rational-polynomial finite tails only when the
/// leading-coefficient ratio lands strictly inside its real domain (`tail > 1`).
/// Other elementary families remain linear-argument only, except for already
/// documented `exp(linear)` compositions.
pub(crate) fn elementary_function_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            let builtin = ctx.builtin_of(fn_id)?;
            match (builtin, args.as_slice()) {
                (BuiltinFn::Log, [base, arg]) => {
                    if let Some(result) =
                        binary_log_argument_limit_at_infinity(ctx, *base, *arg, var, approach)
                    {
                        return Some(result);
                    }
                    if let Some(result) =
                        binary_log_finite_rational_argument_limit_at_infinity(ctx, *base, *arg, var)
                    {
                        return Some(result);
                    }

                    binary_log_linear_exp_argument_limit_at_infinity(
                        ctx, *base, *arg, var, approach,
                    )
                }
                (_, [arg]) => {
                    if let Some(result) = unary_log_finite_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_sqrt_finite_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_abs_finite_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_sqrt_zero_tail_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var, approach,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_cbrt_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var, approach,
                    ) {
                        return Some(result);
                    }
                    if let Some(result) = unary_acosh_finite_rational_argument_limit_at_infinity(
                        ctx, builtin, *arg, var,
                    ) {
                        return Some(result);
                    }
                    unary_elementary_function_limit_at_infinity(ctx, builtin, *arg, var, approach)
                }
                _ => None,
            }
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            elementary_argument_limit_at_infinity(ctx, BuiltinFn::Exp, exp, var, approach)
        }
        _ => None,
    }
}

/// `ln(P(x)) - ln(Q(x))` at +inf collapses the indeterminate inf - inf to
/// `ln(lim P/Q)`: a finite `ln(lc_P/lc_Q)` when the degrees match, `+inf`
/// when P outgrows Q, and `-inf` when Q outgrows P. P and Q must be
/// polynomials tending to +inf (positive leading coefficient) so each
/// logarithm is eventually real. Runs before the generic Sub handling,
/// which would otherwise return the unresolved inf - inf.
pub(super) fn log_difference_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Sub(lhs, rhs) = ctx.get(expr).clone() else {
        return None;
    };
    let p = bare_natural_log_argument(ctx, lhs)?;
    let q = bare_natural_log_argument(ctx, rhs)?;
    let p_growth = polynomial_growth_info(ctx, p, var)?;
    let q_growth = polynomial_growth_info(ctx, q, var)?;
    // Both arguments must tend to +inf so the logarithms are eventually real.
    if !p_growth.leading_coeff.is_positive() || !q_growth.leading_coeff.is_positive() {
        return None;
    }
    use std::cmp::Ordering;
    match p_growth.degree.cmp(&q_growth.degree) {
        Ordering::Equal => {
            let ratio = p_growth.leading_coeff / q_growth.leading_coeff;
            // ln(1) = 0; the raw limit output is not fully simplified, so fold
            // the unit ratio here rather than emit a bare ln(1).
            if ratio.is_one() {
                return Some(ctx.num(0));
            }
            let ratio_expr = ctx.add(Expr::Number(ratio));
            Some(ctx.call_builtin(BuiltinFn::Ln, vec![ratio_expr]))
        }
        Ordering::Greater => Some(mk_infinity(ctx, InfSign::Pos)),
        Ordering::Less => Some(mk_infinity(ctx, InfSign::Neg)),
    }
}

/// `lim_{x->+inf} [Σ cᵢ·ln(pᵢ(x)) + Σ dⱼ·arctan(qⱼ(x))]` for polynomials `pᵢ` (optionally under `abs`)
/// with POSITIVE leading coefficient and polynomials `qⱼ` of degree ≥1. The log block behaves like
/// `s·ln x + Σ cᵢ·ln(lead pᵢ)` with `s = Σ cᵢ·deg pᵢ`, and each `arctan(qⱼ) -> sign(lead qⱼ)·π/2`. So
/// `s>0 -> +inf`, `s<0 -> -inf` (a bounded arctan block cannot offset a divergent log), and `s=0`
/// gives the finite `Σ cᵢ·ln(lead pᵢ) + (Σ dⱼ·sign(lead qⱼ))·π/2`. Generalises the two-term log
/// difference to N terms AND absorbs the arctan terms of a rational partial-fraction antiderivative,
/// so `∫_a^∞ p/q` with an irreducible quadratic factor resolves regardless of Add-tree order (the
/// additive fallback splits the logs individually into `+inf - inf` and stalls).
fn log_sum_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::{One, Signed, Zero};
    if approach != InfSign::Pos {
        return None;
    }
    let mut log_terms: Vec<(BigRational, ExprId)> = Vec::new();
    let mut arctan_terms: Vec<(BigRational, ExprId)> = Vec::new();
    collect_signed_log_and_arctan_terms(
        ctx,
        expr,
        &rational_one(),
        &mut log_terms,
        &mut arctan_terms,
    )?;
    // The log block is the core: at least two log terms (a lone log is ±inf, a pure arctan sum is
    // bounded) are needed for the `+inf - inf` cancellation this rule resolves.
    if log_terms.len() < 2 {
        return None;
    }
    let mut degree_sum = BigRational::zero();
    let mut leading_log_terms: Vec<(BigRational, BigRational)> = Vec::new();
    for (coeff, arg) in &log_terms {
        let growth = polynomial_growth_info(ctx, *arg, var)?;
        // Each argument must tend to +inf so its logarithm is eventually real and divergent.
        if !growth.leading_coeff.is_positive() {
            return None;
        }
        degree_sum += coeff.clone() * BigRational::from_integer(growth.degree.into());
        leading_log_terms.push((coeff.clone(), growth.leading_coeff));
    }
    // Each `arctan(q) -> sign(lead q)·π/2` (q must diverge: degree ≥ 1). Accumulate the π/2 coefficient.
    let mut pi_half_coeff = BigRational::zero();
    for (coeff, arg) in &arctan_terms {
        let growth = polynomial_growth_info(ctx, *arg, var)?;
        let side = if growth.leading_coeff.is_positive() {
            BigRational::one()
        } else {
            -BigRational::one()
        };
        pi_half_coeff += coeff.clone() * side;
    }
    // A divergent log block dominates the bounded arctan block.
    if degree_sum.is_positive() {
        return Some(mk_infinity(ctx, InfSign::Pos));
    }
    if degree_sum.is_negative() {
        return Some(mk_infinity(ctx, InfSign::Neg));
    }
    // s == 0: the finite `Σ cᵢ·ln(lead_i) + (Σ dⱼ·sign(lead qⱼ))·π/2`.
    // Exact zero decision — dropping a true-but-tiny log part would emit a
    // wrong finite value; on `None` (unfactorable) the expression is built,
    // which is value-correct either way.
    let log_is_zero =
        exact_log_combination_is_zero(&leading_log_terms, &BigRational::zero()) == Some(true);
    let log_part = if log_is_zero {
        None
    } else {
        Some(build_log_combination_expr(
            ctx,
            leading_log_terms,
            BigRational::zero(),
        ))
    };
    let pi_coeff = pi_half_coeff / BigRational::from_integer(2.into());
    let pi_part = if pi_coeff.is_zero() {
        None
    } else {
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        if pi_coeff.is_one() {
            Some(pi)
        } else {
            let coeff_expr = ctx.add(Expr::Number(pi_coeff));
            Some(ctx.add(Expr::Mul(coeff_expr, pi)))
        }
    };
    match (log_part, pi_part) {
        (None, None) => Some(ctx.num(0)),
        (Some(l), None) => Some(l),
        (None, Some(p)) => Some(p),
        (Some(l), Some(p)) => Some(ctx.add(Expr::Add(l, p))),
    }
}

/// `lim_{x->+inf} ln(sum c_i b_i^(s_i x)) / (slope x) = ln(max b_i^s_i) / slope`.
/// The dominant exponential sets the growth: with `B = max effective base`,
/// `ln(sum) = ln(B^x (c_dom + o(1))) = x ln B + O(1)`, so the quotient tends to
/// `ln(B) / slope`. Resolves `ln(2^x + 3^x)/x = ln 3`,
/// `ln(2^x + 3^x + 5^x)/x = ln 5`, `ln(5^x - 3^x)/x = ln 5`. Rational bases
/// with integer exponent slopes (so the effective base is rational and
/// comparable exactly); e-bases and mixed sums decline.
pub(super) fn log_exp_sum_dominance_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::{One, Signed, Zero};
    if !matches!(approach, InfSign::Pos) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let arg = bare_natural_log_argument(ctx, num)?;
    // den = slope * x: a degree-1 polynomial with no constant term.
    let den_poly = Polynomial::from_expr(ctx, den, &var_name).ok()?;
    if den_poly.degree() != 1 || !den_poly.coeffs.first().is_some_and(|c| c.is_zero()) {
        return None;
    }
    let den_slope = den_poly.coeffs.get(1).cloned()?;
    if !den_slope.is_positive() {
        return None;
    }
    let mut terms: Vec<(BigRational, BigRational)> = Vec::new();
    collect_rational_exp_terms(ctx, arg, &var_name, &rational_one(), &mut terms)?;
    if terms.is_empty() {
        return None;
    }
    let max_base = terms.iter().map(|(_, b)| b.clone()).max()?;
    // The dominant exponential must actually grow.
    if max_base <= BigRational::one() {
        return None;
    }
    // The dominant terms' coefficients must sum positive, so the sum -> +inf
    // and its logarithm is defined.
    let dominant_sum: BigRational = terms
        .iter()
        .filter(|(_, b)| *b == max_base)
        .map(|(c, _)| c.clone())
        .sum();
    if !dominant_sum.is_positive() {
        return None;
    }
    let base_expr = ctx.add(Expr::Number(max_base));
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base_expr]);
    if den_slope.is_one() {
        Some(ln_base)
    } else {
        let inv = BigRational::one() / den_slope;
        let coeff = ctx.add(Expr::Number(inv));
        Some(ctx.add(Expr::Mul(coeff, ln_base)))
    }
}

pub(crate) fn additive_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_add_limit_results(ctx, lhs_limit, rhs_limit)
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_sub_limit_results(ctx, lhs_limit, rhs_limit)
        }
        _ => None,
    }
}

pub(crate) fn multiplicative_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let inner_limit = try_limit_rules_at_infinity(ctx, inner, var, approach)?;
            Some(negate_limit_result(ctx, inner_limit))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_mul_limit_results(ctx, lhs_limit, rhs_limit)
        }
        Expr::Div(num, den) => {
            let num_limit = try_limit_rules_at_infinity(ctx, num, var, approach)?;
            let den_limit = try_limit_rules_at_infinity(ctx, den, var, approach)?;
            combine_div_limit_results(ctx, num_limit, den_limit)
        }
        _ => None,
    }
}

fn is_eventually_nonzero_expr_at_infinity(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    if constant_rational_value(ctx, expr).is_some_and(|value| !value.is_zero()) {
        return true;
    }
    if known_positive_constant_exceeds_one(ctx, expr) {
        return true;
    }
    if polynomial_argument_tail_sign(ctx, expr, var, approach).is_some() {
        return true;
    }
    if rational_polynomial_argument_tail_sign(ctx, expr, var, approach).is_some() {
        return true;
    }
    if rational_polynomial_argument_zero_tail_sign(ctx, expr, var, approach).is_some() {
        return true;
    }
    rational_polynomial_argument_finite_tail_value(ctx, expr, var)
        .is_some_and(|value| !value.is_zero())
}

fn is_eventually_positive_expr_at_infinity(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    if constant_rational_value(ctx, expr).is_some_and(|value| value.is_positive()) {
        return true;
    }
    if known_positive_constant_exceeds_one(ctx, expr) {
        return true;
    }
    if polynomial_argument_tail_sign(ctx, expr, var, approach) == Some(InfSign::Pos) {
        return true;
    }
    if rational_polynomial_argument_tail_sign(ctx, expr, var, approach) == Some(InfSign::Pos) {
        return true;
    }
    if rational_polynomial_argument_zero_tail_sign(ctx, expr, var, approach) == Some(InfSign::Pos) {
        return true;
    }
    rational_polynomial_argument_finite_tail_value(ctx, expr, var)
        .is_some_and(|value| value.is_positive())
}

fn is_eventually_real_expr_at_infinity(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    if known_positive_constant_exceeds_one(ctx, expr) {
        return true;
    }
    if polynomial_growth_info(ctx, expr, var).is_some() {
        return true;
    }

    match ctx.get(expr) {
        Expr::Variable(_) | Expr::Number(_) => true,
        Expr::Constant(Constant::I | Constant::Infinity | Constant::Undefined) => false,
        Expr::Neg(inner) | Expr::Hold(inner) => {
            is_eventually_real_expr_at_infinity(ctx, *inner, var, approach)
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            is_eventually_real_expr_at_infinity(ctx, *lhs, var, approach)
                && is_eventually_real_expr_at_infinity(ctx, *rhs, var, approach)
        }
        Expr::Div(num, den) => {
            is_eventually_real_expr_at_infinity(ctx, *num, var, approach)
                && is_eventually_real_expr_at_infinity(ctx, *den, var, approach)
                && is_eventually_nonzero_expr_at_infinity(ctx, *den, var, approach)
        }
        Expr::Pow(base, exp) => {
            if let Some((_pow_base, power)) = parse_pow_int(ctx, expr) {
                if power < 0 && !is_eventually_nonzero_expr_at_infinity(ctx, *base, var, approach) {
                    return false;
                }
                return is_eventually_real_expr_at_infinity(ctx, *base, var, approach);
            }
            if constant_rational_value(ctx, *exp).is_some_and(|value| {
                value.denom() == &BigInt::from(2) && value.numer() == &BigInt::from(1)
            }) {
                return is_eventually_positive_expr_at_infinity(ctx, *base, var, approach);
            }
            false
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let arg = args[0];
            match ctx.builtin_of(*fn_id) {
                Some(
                    BuiltinFn::Exp
                    | BuiltinFn::Sin
                    | BuiltinFn::Cos
                    | BuiltinFn::Atan
                    | BuiltinFn::Arctan
                    | BuiltinFn::Tanh
                    | BuiltinFn::Sinh
                    | BuiltinFn::Cosh
                    | BuiltinFn::Asinh
                    | BuiltinFn::Cbrt
                    | BuiltinFn::Abs,
                ) => is_eventually_real_expr_at_infinity(ctx, arg, var, approach),
                Some(BuiltinFn::Sqrt | BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) => {
                    is_eventually_positive_expr_at_infinity(ctx, arg, var, approach)
                }
                Some(BuiltinFn::Acosh) => {
                    unbounded_argument_tail_sign(ctx, arg, var, approach) == Some(InfSign::Pos)
                        || rational_polynomial_argument_finite_tail_value(ctx, arg, var)
                            .is_some_and(|value| value > rational_one())
                        || constant_rational_value(ctx, arg)
                            .is_some_and(|value| value > rational_one())
                }
                _ => false,
            }
        }
        _ => false,
    }
}

pub(super) fn is_bounded_elementary_expr_at_infinity(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    if !depends_on(ctx, expr, var) {
        return is_eventually_real_expr_at_infinity(ctx, expr, var, approach);
    }

    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            matches!(
                ctx.builtin_of(*fn_id),
                Some(
                    BuiltinFn::Sin
                        | BuiltinFn::Cos
                        | BuiltinFn::Atan
                        | BuiltinFn::Arctan
                        | BuiltinFn::Tanh,
                )
            ) && is_eventually_real_expr_at_infinity(ctx, args[0], var, approach)
        }
        Expr::Neg(inner) => is_bounded_elementary_expr_at_infinity(ctx, *inner, var, approach),
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            is_bounded_elementary_expr_at_infinity(ctx, *lhs, var, approach)
                && is_bounded_elementary_expr_at_infinity(ctx, *rhs, var, approach)
        }
        Expr::Div(num, den) => {
            is_bounded_elementary_expr_at_infinity(ctx, *num, var, approach)
                && !depends_on(ctx, *den, var)
                && constant_rational_value(ctx, *den).is_some_and(|value| !value.is_zero())
        }
        _ => false,
    }
}

/// Conservative bounded-over-divergent rule as `x -> ±∞`.
///
/// This is intentionally narrow: only real-domain globally bounded elementary
/// numerators (`sin`/`cos`/`arctan`/`tanh`, plus finite arithmetic combinations
/// of bounded pieces and constants) are accepted, and the denominator must
/// already be proven divergent by the existing infinity rules.
pub(crate) fn bounded_elementary_over_divergent_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    if !depends_on(ctx, num, var)
        || !is_bounded_elementary_expr_at_infinity(ctx, num, var, approach)
    {
        return None;
    }

    let den_limit = try_limit_rules_at_infinity(ctx, den, var, approach)?;
    infinity_sign_of_expr(ctx, den_limit)?;
    Some(ctx.num(0))
}

pub(super) fn bounded_elementary_times_decaying_exp_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() else {
        return None;
    };

    if let Some(exp_info) = scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
        if (exp_info.coeff.is_zero() || exp_info.tail == InfSign::Neg)
            && is_bounded_elementary_expr_at_infinity(ctx, rhs, var, approach)
        {
            return Some(ctx.num(0));
        }
    }

    if let Some(exp_info) = scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
        if (exp_info.coeff.is_zero() || exp_info.tail == InfSign::Neg)
            && is_bounded_elementary_expr_at_infinity(ctx, lhs, var, approach)
        {
            return Some(ctx.num(0));
        }
    }

    None
}

/// Dominance rule for polynomial-argument exponentials against polynomial growth as `x -> ±∞`.
///
/// This is intentionally narrow: only `exp(p(x))`/`e^(p(x))` with optional
/// numeric scaling is compared with polynomials whose relevant leading
/// coefficient is numeric. Parameterized exponent tails and nested
/// exponentials remain residual.
pub(crate) fn exponential_polynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(info) = nonzero_scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, rhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            if let Some(info) = nonzero_scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, lhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(info) = nonzero_scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, rhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            if let Some(info) = nonzero_scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, lhs, var)?;
                return scale_infinity(ctx, &(-info.coeff), InfSign::Pos);
            }
            None
        }
        Expr::Div(num, den) => {
            if let Some(den_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, den, var, approach)
            {
                if den_info.tail == InfSign::Pos {
                    polynomial_or_numeric_tail_sign(ctx, num, var, approach)?;
                    return Some(ctx.num(0));
                }

                if let Some(num_sign) = polynomial_or_numeric_tail_sign(ctx, num, var, approach) {
                    return scale_infinity(
                        ctx,
                        &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                        num_sign,
                    );
                }
            }

            if let Some(num_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, num, var, approach)
            {
                if num_info.tail != InfSign::Pos {
                    return None;
                }
                let den_sign = polynomial_or_numeric_tail_sign(ctx, den, var, approach)?;
                return scale_infinity(ctx, &num_info.coeff, den_sign);
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(info) = scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
                if info.coeff.is_zero() || info.tail == InfSign::Neg {
                    polynomial_growth_info(ctx, rhs, var)?;
                    return Some(ctx.num(0));
                }
            }
            if let Some(info) = scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
                if info.coeff.is_zero() || info.tail == InfSign::Neg {
                    polynomial_growth_info(ctx, lhs, var)?;
                    return Some(ctx.num(0));
                }
            }
            None
        }
        _ => None,
    }
}

/// Dominance rule for domain-safe subpolynomial tails against polynomial growth.
///
/// Logs and inverse hyperbolic cosine accept polynomial and rational-polynomial
/// arguments only when the argument tends to `+∞`, which keeps the real-domain
/// tail explicit. Square roots remain linear-only here; their polynomial
/// arguments can carry polynomial-scale growth. Cube roots and inverse
/// hyperbolic sine are total-real, so their signed linear tails are preserved.
/// Unsupported nonlinear arguments and
/// subpolynomial-vs-subpolynomial comparisons remain residual.
pub(crate) fn subpolynomial_polynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if scaled_subpolynomial_tail_info(ctx, lhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, rhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            if scaled_subpolynomial_tail_info(ctx, rhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, lhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if scaled_subpolynomial_tail_info(ctx, lhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, rhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, neg_inf_sign(sign)));
            }
            if scaled_subpolynomial_tail_info(ctx, rhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, lhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            None
        }
        Expr::Div(num, den) => {
            if scaled_subpolynomial_tail_info(ctx, num, var, approach).is_some() {
                polynomial_growth_info(ctx, den, var)?;
                return Some(ctx.num(0));
            }

            if let Some(den_info) = nonzero_scaled_subpolynomial_tail_info(ctx, den, var, approach)
            {
                let growth = polynomial_growth_info(ctx, num, var)?;
                let num_sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return scale_infinity(
                    ctx,
                    &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                    num_sign,
                );
            }
            None
        }
        _ => None,
    }
}

/// Any positive power of the variable dominates any power of the logarithm:
/// `c ln(x)^a / x^b -> 0` and `c x^b / ln(x)^a -> sign(c) * inf`, for a >= 1
/// integer and b > 0 rational (fractional included). This generalizes the
/// subpolynomial/polynomial rule (single `ln(x)` over an INTEGER power) to
/// higher log powers (`ln(x)^2 / x`) and fractional powers (`ln(x)/sqrt(x)`).
/// Only meaningful as `var -> +inf` (the logarithm needs `x > 0`).
pub(super) fn polylog_power_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    // c ln(x)^a / x^b -> 0 (both coefficients nonzero).
    if let (Some((num_coeff, _)), Some((den_coeff, _))) = (
        constant_times_log_power(ctx, num, var, approach),
        positive_power_tail(ctx, den, var),
    ) {
        if !num_coeff.is_zero() && !den_coeff.is_zero() {
            return Some(ctx.num(0));
        }
    }
    // c x^b / (c' ln(x)^a) -> sign(c / c') * inf.
    if let (Some((num_coeff, _)), Some((den_coeff, _))) = (
        positive_power_tail(ctx, num, var),
        constant_times_log_power(ctx, den, var, approach),
    ) {
        if !num_coeff.is_zero() && !den_coeff.is_zero() {
            let sign = if (num_coeff / den_coeff).is_positive() {
                InfSign::Pos
            } else {
                InfSign::Neg
            };
            return Some(mk_infinity(ctx, sign));
        }
    }
    None
}

/// `ln(arg)` / `log2(arg)` / `log10(arg)` with `arg -> +inf`.
pub(super) fn is_unbounded_log(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> bool {
    matches!(ctx.get(expr), Expr::Function(fn_id, args)
        if args.len() == 1
            && matches!(
                ctx.builtin_of(*fn_id),
                Some(BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10)
            )
            && log_argument_tail_coeff(ctx, args[0], var, approach).is_some_and(|c| c.is_positive()))
}

/// Dominance rule for polynomial-argument exponentials against domain-safe subpolynomial tails.
///
/// Both sides are accepted only through existing tail analyzers: exponential
/// exponents must have a non-parametric polynomial tail; logarithms and inverse
/// hyperbolic cosine may carry positive polynomial tails; square-root arguments
/// must tend to `+∞` on a linear real tail; cube roots and inverse hyperbolic
/// sine may carry either signed linear tail. Parametric or nested exponential
/// tails remain residual.
pub(crate) fn exponential_subpolynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(exp_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, lhs, var, approach)
            {
                if exp_info.tail == InfSign::Pos
                    && scaled_subpolynomial_tail_info(ctx, rhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            if let Some(exp_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, rhs, var, approach)
            {
                if exp_info.tail == InfSign::Pos
                    && scaled_subpolynomial_tail_info(ctx, lhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(exp_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, lhs, var, approach)
            {
                if exp_info.tail == InfSign::Pos
                    && scaled_subpolynomial_tail_info(ctx, rhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            if let Some(exp_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, rhs, var, approach)
            {
                if exp_info.tail == InfSign::Pos
                    && scaled_subpolynomial_tail_info(ctx, lhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &(-exp_info.coeff), InfSign::Pos);
                }
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(exp_info) = scaled_polynomial_exp_tail_info(ctx, lhs, var, approach) {
                if let Some(subpoly_info) = scaled_subpolynomial_tail_info(ctx, rhs, var, approach)
                {
                    return scaled_exp_subpoly_product_limit(ctx, exp_info, subpoly_info);
                }
            }
            if let Some(exp_info) = scaled_polynomial_exp_tail_info(ctx, rhs, var, approach) {
                if let Some(subpoly_info) = scaled_subpolynomial_tail_info(ctx, lhs, var, approach)
                {
                    return scaled_exp_subpoly_product_limit(ctx, exp_info, subpoly_info);
                }
            }
            None
        }
        Expr::Div(num, den) => {
            if let Some(den_info) = nonzero_scaled_polynomial_exp_tail_info(ctx, den, var, approach)
            {
                if let Some(num_info) = scaled_subpolynomial_tail_info(ctx, num, var, approach) {
                    let Some(num_sign) = subpolynomial_tail_sign(&num_info) else {
                        return Some(ctx.num(0));
                    };
                    if den_info.tail == InfSign::Pos {
                        return Some(ctx.num(0));
                    }
                    return scale_infinity(
                        ctx,
                        &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                        num_sign,
                    );
                }
            }

            if let Some(num_info) = scaled_polynomial_exp_tail_info(ctx, num, var, approach) {
                if let Some(den_info) =
                    nonzero_scaled_subpolynomial_tail_info(ctx, den, var, approach)
                {
                    if num_info.coeff.is_zero() || num_info.tail == InfSign::Neg {
                        return Some(ctx.num(0));
                    }
                    return scale_infinity(ctx, &(num_info.coeff / den_info.coeff), InfSign::Pos);
                }
            }
            None
        }
        _ => None,
    }
}

/// Polynomial limit rule for `P(x)` as `x -> ±∞`.
///
/// This handles polynomial expressions whose leading coefficient in `var` is a
/// numeric constant. Symbolic leading coefficients remain unresolved because
/// their sign is not known under the conservative limit policy.
pub(crate) fn polynomial_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
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
        return Some(ctx.num(0));
    }

    let var_idx = poly.var_index(var_name)?;
    let degree = poly.degree_in(var_idx);
    if degree == 0 {
        return None;
    }

    let leading_coeff = poly.leading_coeff_in(var_idx);
    let leading_value = leading_coeff.constant_value()?;
    let sign = limit_growth_sign(&leading_value, degree, approach);
    Some(mk_infinity(ctx, sign))
}

/// Try all limit-at-infinity rules in conservative order.
///
/// Order:
/// 1. Constant
/// 2. Variable
/// 3. Power
/// 4. Reciprocal power
/// 5. Elementary exact-argument functions
/// 6. Additive combination
/// 7. Determinate multiplicative combination
/// 8. Bounded trig over divergent denominator
/// 9. Square-root polynomial ratio with matching growth
/// 10. Polynomial over square-root polynomial with matching growth
/// 11. Absolute-value polynomial ratio with matching growth
/// 12. Polynomial over absolute-value polynomial with matching growth
/// 13. Exact exponential-vs-polynomial dominance
/// 14. Exact subpolynomial-vs-polynomial dominance
/// 15. Exact exponential-vs-subpolynomial dominance
/// 16. Polynomial
/// 17. Rational polynomial
pub(crate) fn try_limit_rules_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if let Some(r) = apply_static_empty_real_domain_rule(ctx, expr, var) {
        return Some(r);
    }
    if let Some(r) = apply_constant_rule(ctx, expr, var) {
        return Some(r);
    }
    if let Some(r) = apply_variable_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_power_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_rational_power_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_same_base_power_quotient_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = exp_sum_top_level_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_reciprocal_power_rule(ctx, expr, var) {
        return Some(r);
    }
    if let Some(r) = one_to_infinity_power_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = general_base_exponential_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = constant_base_exponential_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = inf_to_zero_power_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = elementary_function_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = log_exp_sum_dominance_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = log_difference_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = log_sum_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = product_log_unit_argument_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = additive_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = multiplicative_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = bounded_elementary_over_divergent_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) =
        bounded_elementary_times_decaying_exp_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = sqrt_quadratic_minus_linear_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = sqrt_minus_sqrt_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = radical_conjugate_product_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = cbrt_conjugate_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = nth_root_conjugate_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = sqrt_polynomial_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = polynomial_sqrt_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = abs_polynomial_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = polynomial_abs_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = exponential_polynomial_dominance_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = subpolynomial_polynomial_dominance_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = polylog_power_dominance_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = exponential_subpolynomial_dominance_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = polynomial_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = bounded_noise_rational_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = rational_difference_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = exp_sum_quotient_dominance_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = general_exp_vs_polynomial_dominance_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = polynomial_times_decaying_exponential_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    rational_poly_limit(ctx, expr, var, approach)
}

/// A polynomial times a DECAYING general-base exponential at infinity tends to
/// 0: the exponential decay outruns any polynomial growth. Covers
/// `x * 2^(-x) -> 0` (at +inf) and `x * 2^x -> 0` (at -inf). The product is the
/// 0 * inf form the generic Mul branch leaves residual; here the 0 factor is a
/// genuine exponential, so the product is 0 regardless of the polynomial sign.
pub(super) fn polynomial_times_decaying_exponential_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::Zero;
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() else {
        return None;
    };
    for (poly_cand, exp_cand) in [(lhs, rhs), (rhs, lhs)] {
        let is_polynomial =
            Polynomial::from_expr(ctx, poly_cand, &var_name).is_ok_and(|p| p.degree() >= 1);
        // The exponential factor: any Pow with a var-free base (rational or
        // provable like π); the general-base rule below decides its decay.
        let is_constant_base_pow = matches!(
            ctx.get(exp_cand),
            Expr::Pow(base, _) if !depends_on(ctx, *base, var)
        );
        if !is_polynomial || !is_constant_base_pow {
            continue;
        }
        // The exponential factor must DECAY (limit 0) so it dominates the poly.
        let exp_limit = general_base_exponential_limit_at_infinity(ctx, exp_cand, var, approach)?;
        if crate::numeric_eval::as_rational_const(ctx, exp_limit).is_some_and(|v| v.is_zero()) {
            return Some(ctx.num(0));
        }
    }
    None
}

/// A quotient of a general-base exponential sum against a polynomial at
/// infinity: an exponential `b^x` with `b > 1` beats any polynomial, so
/// `(sum b_i^x) / poly -> +-inf` and `poly / (sum b_i^x) -> 0`. The engine
/// already settles the natural base; this covers `2^x / x^2 -> +inf`,
/// `x^10 / 2^x -> 0`. Reuses the effective-base parser for the exponential
/// side and the polynomial reader for the other.
pub(super) fn general_exp_vs_polynomial_dominance_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if !matches!(approach, InfSign::Pos) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    // Numerator is an exponential sum, denominator a polynomial: exp wins.
    if let Some(num_sign) = exp_sum_dominant_sign(ctx, num, &var_name) {
        if let Some(den_sign) = positive_degree_polynomial_tail_sign(ctx, den, &var_name) {
            let positive = num_sign == den_sign;
            return Some(mk_infinity(
                ctx,
                if positive { InfSign::Pos } else { InfSign::Neg },
            ));
        }
    }
    // Numerator is a polynomial, denominator an exponential sum: poly loses.
    if positive_degree_polynomial_tail_sign(ctx, num, &var_name).is_some()
        && exp_sum_dominant_sign(ctx, den, &var_name).is_some()
    {
        return Some(ctx.num(0));
    }
    None
}

/// La SUMA de exponenciales de base racional a NIVEL RAÍZ en `+∞`, decidida por
/// la base dominante. La maquinaria (`exp_sum_dominant_sign`) existía y solo se
/// consumía DENTRO de cocientes: `limit(3^x − 2^x, x, ∞)` declinaba a pesar de
/// que la misma suma como denominador ya se decidía. Guardas: dos o más
/// términos (el término único tiene dueño propio y robarlo movería sus pasos),
/// solo `+∞` (en `−∞` las bases >1 decaen y ese camino ya funciona), y las
/// bases e/π quedan para la generalización nombrada de `collect_rational_exp_terms`
/// a pares provables. `2^x − 2^x` (dominante con suma cero) declina honesto.
pub(super) fn exp_sum_top_level_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if approach != InfSign::Pos {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();

    let mut terms: Vec<(BigRational, BigRational)> = Vec::new();
    collect_rational_exp_terms(ctx, expr, &var_name, &rational_one(), &mut terms)?;
    if terms.len() < 2 {
        return None;
    }
    let sign = exp_sum_dominant_sign(ctx, expr, &var_name)?;
    Some(mk_infinity(
        ctx,
        if sign > 0 { InfSign::Pos } else { InfSign::Neg },
    ))
}

/// A quotient of sums of rational-base exponentials at infinity, decided by
/// the dominant base: `(sum c_i b_i^x) / (sum d_j e_j^x)` tends to the ratio
/// of the dominant coefficients when the top bases match, to `+-inf` when the
/// numerator's dominant base is larger, and to 0 when it is smaller. Reuses the
/// effective-base term parser. Resolves `3^x/2^x -> +inf`,
/// `(2^x+3^x)/3^x -> 1`, `(3^x-2^x)/(3^x+2^x) -> 1`.
pub(super) fn exp_sum_quotient_dominance_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::{Signed, Zero};
    if !matches!(approach, InfSign::Pos) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let mut num_terms: Vec<(BigRational, BigRational)> = Vec::new();
    let mut den_terms: Vec<(BigRational, BigRational)> = Vec::new();
    collect_rational_exp_terms(ctx, num, &var_name, &rational_one(), &mut num_terms)?;
    collect_rational_exp_terms(ctx, den, &var_name, &rational_one(), &mut den_terms)?;
    // Require at least one genuinely growing exponential on each side so this
    // is an exponential quotient, not a rational one (owned upstream).
    let one = BigRational::one();
    if !num_terms.iter().any(|(_, b)| *b > one) || !den_terms.iter().any(|(_, b)| *b > one) {
        return None;
    }
    let num_max = num_terms.iter().map(|(_, b)| b.clone()).max()?;
    let den_max = den_terms.iter().map(|(_, b)| b.clone()).max()?;
    let dominant_sum = |terms: &[(BigRational, BigRational)], base: &BigRational| -> BigRational {
        terms
            .iter()
            .filter(|(_, b)| b == base)
            .map(|(c, _)| c.clone())
            .sum()
    };
    let num_dom = dominant_sum(&num_terms, &num_max);
    let den_dom = dominant_sum(&den_terms, &den_max);
    // A cancelled dominant coefficient leaves a lower true dominant - too subtle
    // here, so decline.
    if den_dom.is_zero() {
        return None;
    }
    use std::cmp::Ordering;
    match num_max.cmp(&den_max) {
        Ordering::Less => Some(ctx.num(0)),
        Ordering::Equal => {
            let ratio = &num_dom / &den_dom;
            Some(ctx.add(Expr::Number(ratio)))
        }
        Ordering::Greater => {
            if num_dom.is_zero() {
                return None;
            }
            let positive = num_dom.is_positive() == den_dom.is_positive();
            Some(mk_infinity(
                ctx,
                if positive { InfSign::Pos } else { InfSign::Neg },
            ))
        }
    }
}

/// The exponential indeterminate form `1^inf` at infinity: when the base
/// tends to 1 and the exponent diverges, `base^exp = exp(exp * ln(base))` and,
/// since `ln(1 + h) ~ h` for `h = base - 1 -> 0`, the limit is
/// `exp(lim exp * (base - 1))`. Resolves `(1 + a/x)^x -> e^a`,
/// `(1 + 1/x)^(2x) -> e^2`, `(1 + 1/x^2)^x -> 1`, `((2x+1)/(2x-1))^x -> e`.
/// Gated on a unit base limit and a divergent exponent, so `2^x`, `x^x`, and
/// any base that does not tend to 1 are left untouched.
pub(super) fn one_to_infinity_power_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::One;
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    // Both the base and the exponent must carry the variable for an
    // indeterminate 1^inf; a constant base or exponent is a different form.
    if !depends_on(ctx, base, var) || !depends_on(ctx, exp, var) {
        return None;
    }
    // The base must tend to exactly 1.
    let base_limit = try_limit_rules_at_infinity(ctx, base, var, approach)?;
    if !crate::numeric_eval::as_rational_const(ctx, base_limit).is_some_and(|v| v.is_one()) {
        return None;
    }
    // The exponent must diverge (otherwise 1^finite = 1 is the continuous case).
    let exp_limit = try_limit_rules_at_infinity(ctx, exp, var, approach)?;
    limit_value_infinite_sign(ctx, exp_limit)?;
    // L = lim exp * (base - 1), rationalized so the quotient limit can read it.
    let one = ctx.num(1);
    let h = ctx.add(Expr::Sub(base, one));
    let product = ctx.add(Expr::Mul(exp, h));
    let (num, den) = rationalize_to_fraction(ctx, product);
    let combined = ctx.add(Expr::Div(num, den));
    let l_limit = try_limit_rules_at_infinity(ctx, combined, var, approach)?;
    // exp(L): e^finite, e^(+inf) = inf, e^(-inf) = 0.
    match limit_value_infinite_sign(ctx, l_limit) {
        Some(1) => Some(mk_infinity(ctx, InfSign::Pos)),
        Some(_) => Some(ctx.num(0)),
        None => {
            // Fold the degenerate finite exponents: e^0 = 1, e^1 = e.
            if let Some(value) = crate::numeric_eval::as_rational_const(ctx, l_limit) {
                use num_traits::{One, Zero};
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

/// The `inf * 0` form `g(x) * ln(f(x))` at infinity when `f -> 1` (so
/// `ln f -> 0`) and `g` diverges. Since `ln(1 + h) ~ h` for `h = f - 1 -> 0`,
/// the limit is `lim g * (f - 1)` -- the same `L` the 1^inf power rule computes,
/// exposed for the standalone product. Resolves `x ln(1 + a/x) -> a`,
/// `x ln((x+1)/x) -> 1`, `x^2 ln(1 + 1/x^2) -> 1`, `x ln(1 - 1/x) -> -1`. The
/// `f -> 1` gate keeps it off `x ln(x)` (`f -> inf`) and `ln(2) x` (`f` constant),
/// and the divergent-cofactor gate leaves the continuous `finite * 0` case alone.
pub(super) fn product_log_unit_argument_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::One;
    let Expr::Mul(a, b) = ctx.get(expr).clone() else {
        return None;
    };
    let natural_log_arg = |ctx: &Context, e: ExprId| -> Option<ExprId> {
        if let Expr::Function(fn_id, args) = ctx.get(e) {
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Ln) {
                return Some(args[0]);
            }
        }
        None
    };
    let (cofactor, ln_arg) = if let Some(arg) = natural_log_arg(ctx, a) {
        (b, arg)
    } else if let Some(arg) = natural_log_arg(ctx, b) {
        (a, arg)
    } else {
        return None;
    };
    if !depends_on(ctx, cofactor, var) || !depends_on(ctx, ln_arg, var) {
        return None;
    }
    // The log argument must tend to exactly 1 (so ln -> 0): a genuine inf*0.
    let arg_limit = try_limit_rules_at_infinity(ctx, ln_arg, var, approach)?;
    if !crate::numeric_eval::as_rational_const(ctx, arg_limit).is_some_and(|v| v.is_one()) {
        return None;
    }
    // The cofactor must diverge (otherwise finite * 0 = 0 is the continuous case).
    let cofactor_limit = try_limit_rules_at_infinity(ctx, cofactor, var, approach)?;
    limit_value_infinite_sign(ctx, cofactor_limit)?;
    // L = lim g * (f - 1), rationalized so the quotient limit can read it.
    let one = ctx.num(1);
    let h = ctx.add(Expr::Sub(ln_arg, one));
    let product = ctx.add(Expr::Mul(cofactor, h));
    let (num, den) = rationalize_to_fraction(ctx, product);
    let combined = ctx.add(Expr::Div(num, den));
    try_limit_rules_at_infinity(ctx, combined, var, approach)
}

/// The `1^inf` form at a FINITE point: `(1 + x)^(1/x) -> e` and friends. Same
/// identity as the infinity case - `exp(lim exp*(base-1))` via `ln(1+h) ~ h` -
/// but the exponent (e.g. `1/x` at 0) has no signed bilateral limit, so the
/// gate is on the PRODUCT instead: a unit base limit and a NONZERO product
/// limit `L = lim exp*(base-1)` (a nonzero L forces the exponent to diverge,
/// so the form is genuinely indeterminate; a continuous `1^finite` gives
/// `L = 0` and is left to ordinary substitution).
pub(super) fn apply_finite_one_to_infinity_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    use num_traits::{One, Zero};
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    if !depends_on(ctx, base, var) || !depends_on(ctx, exp, var) {
        return None;
    }
    let base_limit = try_limit_rules_at_finite(ctx, base, var, point)?;
    if !crate::numeric_eval::as_rational_const(ctx, base_limit).is_some_and(|v| v.is_one()) {
        return None;
    }
    let one = ctx.num(1);
    let h = ctx.add(Expr::Sub(base, one));
    let product = ctx.add(Expr::Mul(exp, h));
    let (num, den) = rationalize_to_fraction(ctx, product);
    let combined = ctx.add(Expr::Div(num, den));
    let l_limit = try_limit_rules_at_finite(ctx, combined, var, point)?;
    // A zero product limit is the continuous 1^finite case (1^c = 1), owned by
    // ordinary substitution; only fire on the genuinely indeterminate form.
    if crate::numeric_eval::as_rational_const(ctx, l_limit).is_some_and(|v| v.is_zero()) {
        return None;
    }
    match limit_value_infinite_sign(ctx, l_limit) {
        Some(1) => Some(mk_infinity(ctx, InfSign::Pos)),
        Some(_) => Some(ctx.num(0)),
        None => {
            if crate::numeric_eval::as_rational_const(ctx, l_limit).is_some_and(|v| v.is_one()) {
                return Some(ctx.add(Expr::Constant(Constant::E)));
            }
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, l_limit)))
        }
    }
}

/// A general-base exponential `b^(s x)` at infinity (`b` a var-free positive
/// constant — rational, or provable like `e/π`): `b^x -> +inf` for `b > 1`,
/// `-> 0` for `0 < b < 1`, `-> 1` for `b = 1`, with the sign of the exponent
/// slope and the approach direction combined. The engine already grows
/// `e^x`; this covers rational bases (`2^x -> inf`, feeding `2^x + 3^x` into
/// the inf^0 rule) and provable constant bases (`(e/π)^x -> 0`).
fn general_base_exponential_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::{Signed, Zero};
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Pow(base, exponent) = ctx.get(expr).clone() else {
        return None;
    };
    if depends_on(ctx, base, var) {
        return None;
    }
    let base_vs_one = constant_base_vs_one(ctx, base)?;
    if base_vs_one == std::cmp::Ordering::Equal {
        return Some(ctx.num(1));
    }
    // The exponent must be linear in x; its sign at the approach decides growth.
    let exp_poly = Polynomial::from_expr(ctx, exponent, &var_name).ok()?;
    if exp_poly.degree() != 1 {
        return None;
    }
    let slope = exp_poly.coeffs.get(1).cloned()?;
    if slope.is_zero() {
        return None;
    }
    // The exponent tends to +inf when its slope and the approach agree in sign.
    let exponent_to_pos_inf = slope.is_positive() == matches!(approach, InfSign::Pos);
    let base_gt_one = base_vs_one == std::cmp::Ordering::Greater;
    if base_gt_one == exponent_to_pos_inf {
        Some(mk_infinity(ctx, InfSign::Pos))
    } else {
        Some(ctx.num(0))
    }
}

/// A quotient of same-exponent exponentials `a^u / b^u` with constant bases:
/// combine into `(a/b)^u` and classify via the general constant-base rule.
/// Closes `e^x / π^x -> 0` (e/π < 1 by exact bounds) without a global
/// simplifier rewrite. Both bases must be INDIVIDUALLY provably positive —
/// `(-2)^x / (-3)^x` must never become `(2/3)^x`, since the original is not
/// even real-valued along the reals.
fn constant_base_exponential_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let (num_base, num_exp) = constant_exponential_atom(ctx, num, var)?;
    let (den_base, den_exp) = constant_exponential_atom(ctx, den, var)?;
    if !structurally_equal_expr(ctx, num_exp, den_exp) {
        return None;
    }
    if !provably_positive_constant_base(ctx, num_base)
        || !provably_positive_constant_base(ctx, den_base)
    {
        return None;
    }
    let combined_base = ctx.add(Expr::Div(num_base, den_base));
    let combined = ctx.add(Expr::Pow(combined_base, num_exp));
    general_base_exponential_limit_at_infinity(ctx, combined, var, approach)
}

/// The `inf^0` form at infinity: when the base diverges to `+inf` and the
/// exponent tends to 0, `base^exp = exp(exp * ln base)` and the limit is
/// `exp(lim exp * ln base)`. With the log-of-exponential-sum dominance rule
/// feeding the inner limit, this resolves `(2^x + 3^x)^(1/x) = e^(ln 3) = 3`.
/// A positive-infinite base keeps ln real; `x^(1/x) = 1` falls out too.
pub(super) fn inf_to_zero_power_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use num_traits::Zero;
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    if !depends_on(ctx, base, var) || !depends_on(ctx, exp, var) {
        return None;
    }
    // base -> +inf keeps it positive (ln real); exp -> 0 makes it inf^0.
    let base_limit = try_limit_rules_at_infinity(ctx, base, var, approach)?;
    if limit_value_infinite_sign(ctx, base_limit) != Some(1) {
        return None;
    }
    let exp_limit = try_limit_rules_at_infinity(ctx, exp, var, approach)?;
    if !crate::numeric_eval::as_rational_const(ctx, exp_limit).is_some_and(|v| v.is_zero()) {
        return None;
    }
    // L = lim exp * ln(base), rationalized into a single fraction and
    // presimplified (drops the unit factor the rationalizer leaves) so the
    // log-of-exponential-sum dominance rule sees a BARE ln in the numerator.
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let product = ctx.add(Expr::Mul(exp, ln_base));
    let (num, den) = rationalize_to_fraction(ctx, product);
    let combined = ctx.add(Expr::Div(num, den));
    let combined = presimplify_safe_for_limit(ctx, combined);
    let l_limit = try_limit_rules_at_infinity(ctx, combined, var, approach)?;
    exp_of_limit_value(ctx, l_limit)
}

/// An additive combination of rational functions at infinity that the
/// per-term additive rule leaves as `inf - inf` (e.g. `(x^2+1)/(x+1) - x`):
/// place the terms over a common denominator and reuse the rational quotient
/// limit. Runs only after the structural additive rules decline, so a sum of
/// terms that each have a finite limit keeps its existing trace. Non-rational
/// operands (sqrt, sin, ...) make the multipoly conversion fail, so the
/// conjugate/elementary paths keep their forms.
pub(super) fn rational_difference_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let (lhs, rhs, subtract) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => (lhs, rhs, true),
        Expr::Add(lhs, rhs) => (lhs, rhs, false),
        _ => return None,
    };
    let (a_num, a_den) = rational_numerator_denominator(ctx, lhs);
    let (b_num, b_den) = rational_numerator_denominator(ctx, rhs);
    // (a_num/a_den) +/- (b_num/b_den) = (a_num b_den +/- b_num a_den)/(a_den b_den).
    let left_cross = ctx.add(Expr::Mul(a_num, b_den));
    let right_cross = ctx.add(Expr::Mul(b_num, a_den));
    let numerator = if subtract {
        ctx.add(Expr::Sub(left_cross, right_cross))
    } else {
        ctx.add(Expr::Add(left_cross, right_cross))
    };
    let denominator = ctx.add(Expr::Mul(a_den, b_den));
    let combined = ctx.add(Expr::Div(numerator, denominator));
    rational_poly_limit(ctx, combined, var, approach)
}

/// A rational quotient at infinity where one or both sides are a polynomial
/// plus BOUNDED additive noise (e.g. `(x + sin x)/x -> 1`): the polynomial
/// part sets the growth, the bounded noise is dominated. Reuses
/// polynomial_growth_info_with_bounded_additive_noise and compares degrees
/// exactly as the pure rational rule does. Gated to fire only when at least
/// one side is NOT a pure polynomial, so rational_poly_limit keeps ownership
/// of the exact poly/poly case.
pub(super) fn bounded_noise_rational_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    if polynomial_growth_info(ctx, num, var).is_some()
        && polynomial_growth_info(ctx, den, var).is_some()
    {
        return None;
    }
    let num_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, num, var, approach)?;
    let den_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, den, var, approach)?;
    if den_growth.leading_coeff.is_zero() {
        return None;
    }
    if num_growth.degree < den_growth.degree {
        return Some(ctx.num(0));
    }
    let ratio = num_growth.leading_coeff / den_growth.leading_coeff;
    if num_growth.degree == den_growth.degree {
        return Some(ctx.add(Expr::Number(ratio)));
    }
    let sign = limit_growth_sign(&ratio, num_growth.degree - den_growth.degree, approach);
    Some(mk_infinity(ctx, sign))
}

/// Evaluate a limit at infinity using conservative rules.
///
/// This runs optional safe pre-simplification, applies direct rules in order,
/// and otherwise returns a residual `limit(...)` expression.
pub fn eval_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
    opts: &LimitOptions,
) -> LimitEvalOutcome {
    // Domain kill-switch (Fase 3 · F0): every rule below reasons with the real
    // order, so it must not run at all outside the real domain — under a
    // complex value domain it FABRICATES (`e^(-1/z²) → 0`, `z·sin(1/z) → 0`,
    // both nonexistent in ℂ), and a real-domain point containing the imaginary
    // unit has no real neighbourhood to reason about (`tanh` at `iπ/2` is a
    // pole that direct substitution turns into a value). Both cases decline to
    // the honest residual BEFORE any rule sees the expression; complex-domain
    // capability is re-granted selectively with analytic justification (F10/F11).
    if opts.complex_enabled {
        // F11: selective re-grant for ANALYTIC shapes (exact meromorphy gate);
        // everything else keeps the F0 kill-switch residual.
        if let Some(value) = try_complex_limit_selective(ctx, expr, var, approach) {
            return LimitEvalOutcome {
                expr: value,
                warning: None,
            };
        }
        let residual = mk_limit_for_approach(ctx, expr, var, approach);
        return LimitEvalOutcome {
            expr: residual,
            warning: Some(COMPLEX_DOMAIN_LIMIT_UNSUPPORTED_WARNING.to_string()),
        };
    }
    if let Approach::Finite(point) | Approach::FiniteOneSided(point, _) = approach {
        if crate::numeric_eval::expr_contains_imaginary(ctx, point) {
            let residual = mk_limit_for_approach(ctx, expr, var, approach);
            return LimitEvalOutcome {
                expr: residual,
                warning: Some(IMAGINARY_POINT_LIMIT_UNSUPPORTED_WARNING.to_string()),
            };
        }
    }

    // Matrix limits are componentwise, all-or-nothing (P0 2026-07-19: the
    // constant rule used to assert `[[1/x,0],[0,1]]` as its OWN limit because
    // `depends_on` skipped matrix entries). Runs after the domain guard so
    // complex/imaginary-point declines stay whole-matrix residuals.
    if let Expr::Matrix { rows, cols, data } = ctx.get(expr) {
        let (rows, cols, data) = (*rows, *cols, data.clone());
        return eval_limit_matrix_componentwise(ctx, expr, rows, cols, data, var, approach, opts);
    }

    let simplified_expr = match opts.presimplify {
        PreSimplifyMode::Off => expr,
        PreSimplifyMode::Safe => presimplify_safe_for_limit(ctx, expr),
    };

    if let Approach::Finite(point) = approach {
        if let Some(result_expr) = try_limit_rules_at_finite(ctx, simplified_expr, var, point) {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
        // Bilateral combiner (frontier-audit limits gap): when the direct
        // bilateral rules decline but BOTH one-sided limits compute, combine
        // them — agreeing ±∞ is the signed divergence (`1/x³` at 0⁺/0⁻ was
        // residual while `1/x²` worked), disagreeing laterals mean the
        // bilateral limit DOES NOT EXIST (`1/x`, `|x|/x` at 0 → undefined
        // with the laterals quoted in the warning). Conservative by
        // construction: any lateral that is residual/undefined/unclassifiable
        // (domain boundaries like `√x`/`ln x` at 0, oscillation `sin(1/x)`)
        // keeps the honest bilateral residual below.
        if let Some(outcome) =
            try_bilateral_limit_from_lateral_agreement(ctx, simplified_expr, var, point)
        {
            return outcome;
        }
        // DNE by oscillation (the frontier-audit item narrowed 2026-07-18 to
        // exactly this): the laterals combiner above declines on `sin(1/x)`
        // because the trig lateral itself never computes — but the INNER
        // argument's divergence is provable, and that alone decides.
        if let Some(outcome) = try_dne_by_oscillation(ctx, simplified_expr, var, point) {
            return outcome;
        }
    }
    if let Approach::FiniteOneSided(point, side) = approach {
        if let Some(result_expr) =
            try_limit_rules_at_finite_one_sided(ctx, simplified_expr, var, point, side)
        {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
    }

    if let Some(sign) = approach.inf_sign() {
        if let Some(result_expr) = try_limit_rules_at_infinity(ctx, simplified_expr, var, sign) {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
        // Reciprocal-substitution fallback: `lim_{x→±∞} g(x) = lim_{u→0±} g(1/u)` (exact — `u = 1/x`
        // approaches 0 from the matching side). Only reached when the direct ∞ rules declined, so it
        // can RESOLVE a previously-residual limit but never overrides a direct result.
        if let Some(result_expr) =
            try_limit_at_infinity_by_reciprocal_substitution(ctx, simplified_expr, var, sign)
        {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
    }

    let residual = mk_limit_for_approach(ctx, simplified_expr, var, approach);
    let warning = match approach {
        Approach::Finite(point) => finite_residual_warning(ctx, simplified_expr, var, point),
        Approach::FiniteOneSided(_, _) => {
            "One-sided finite point limits are not supported safely for this expression yet"
                .to_string()
        }
        Approach::PosInfinity | Approach::NegInfinity => {
            "Could not determine limit safely".to_string()
        }
    };
    LimitEvalOutcome {
        expr: residual,
        warning: Some(warning),
    }
}

/// Evaluate `lim_{x→±∞} expr` by the exact change of variable `u = 1/x`: the limit equals
/// `lim_{u→0±} expr[x ↦ 1/u]` (`u → 0⁺` for `x → +∞`, `u → 0⁻` for `x → −∞`). Reusing the SAME
/// variable for `u` avoids introducing a fresh symbol. Closes notable forms the direct ∞ rules miss,
/// e.g. `x·sin(1/x) → 1` and `(1 + a/x)^x → e^a`. Sound because the substitution is an exact identity
/// and the finite one-sided evaluator it delegates to is itself sound.
pub(super) fn try_limit_at_infinity_by_reciprocal_substitution(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    sign: InfSign,
) -> Option<ExprId> {
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, var));
    let substituted = cas_ast::substitute_expr_by_id(ctx, expr, var, reciprocal);
    // The substitution leaves nested reciprocals and unit factors (`1/(1/x)`, `(1·sin(x))/x`);
    // normalize so the finite evaluator sees the reduced form (`x·sin(1/x)` → `sin(x)/x`). The
    // finite one-sided evaluator re-checks the domain, so canonical normalization is safe here.
    // The substitution leaves nested reciprocals and unit factors (`1/(1/x)`, `sin(x)·(1/x)`) that
    // the cas_math normalizers do not reduce; clean them so the finite evaluator sees `sin(x)/x`.
    let substituted = reduce_reciprocal_substitution_artifacts(ctx, substituted);
    let zero = ctx.num(0);
    let side = match sign {
        InfSign::Pos => FiniteLimitSide::Right,
        InfSign::Neg => FiniteLimitSide::Left,
    };
    try_limit_rules_at_finite_one_sided(ctx, substituted, var, zero, side)
}

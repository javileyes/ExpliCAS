//! `limits_support`: familia `sign_zero`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

/// Determine resulting infinity sign from approach sign and exponent parity.
pub(crate) fn limit_sign(approach: InfSign, power: i64) -> InfSign {
    match approach {
        InfSign::Pos => InfSign::Pos,
        InfSign::Neg => {
            if power % 2 == 0 {
                InfSign::Pos // (-∞)^even = +∞
            } else {
                InfSign::Neg // (-∞)^odd = -∞
            }
        }
    }
}

/// Flatten an additive tree into signed leaf terms (handles Add, Sub,
/// and unary Neg).
pub(super) fn collect_signed_add_terms(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            collect_signed_add_terms(ctx, l, positive, terms);
            collect_signed_add_terms(ctx, r, positive, terms);
        }
        Expr::Sub(l, r) => {
            collect_signed_add_terms(ctx, l, positive, terms);
            collect_signed_add_terms(ctx, r, !positive, terms);
        }
        Expr::Neg(inner) => collect_signed_add_terms(ctx, inner, !positive, terms),
        _ => terms.push((expr, positive)),
    }
}

pub(super) fn finite_endpoint_argument_zero_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    if let Ok(polynomial) = Polynomial::from_expr(ctx, expr, var_name) {
        let (order, derivative) =
            finite_polynomial_local_order_and_derivative(&polynomial, point_value)?;
        if order == 0 {
            return None;
        }
        return finite_local_tail_sign(&derivative, order, side);
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let denominator_value = denominator.eval(point_value);
    if denominator_value.is_zero() {
        return None;
    }

    let (numerator_order, numerator_derivative) =
        finite_polynomial_local_order_and_derivative(&numerator, point_value)?;
    if numerator_order == 0 {
        return None;
    }

    let numerator_tail = finite_local_tail_sign(&numerator_derivative, numerator_order, side)?;
    let denominator_tail = if denominator_value.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    Some(if numerator_tail == denominator_tail {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

pub(super) fn finite_local_tail_positive_on_both_sides(
    derivative_value: &BigRational,
    order: usize,
) -> Option<bool> {
    Some(
        finite_local_tail_sign(derivative_value, order, FiniteLimitSide::Left)? == InfSign::Pos
            && finite_local_tail_sign(derivative_value, order, FiniteLimitSide::Right)?
                == InfSign::Pos,
    )
}

pub(super) fn finite_local_tail_negative_on_both_sides(
    derivative_value: &BigRational,
    order: usize,
) -> Option<bool> {
    Some(
        finite_local_tail_sign(derivative_value, order, FiniteLimitSide::Left)? == InfSign::Neg
            && finite_local_tail_sign(derivative_value, order, FiniteLimitSide::Right)?
                == InfSign::Neg,
    )
}

pub(super) fn finite_endpoint_argument_zero_tail_positive_on_both_sides(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
) -> Option<bool> {
    Some(
        finite_endpoint_argument_zero_tail_sign(
            ctx,
            expr,
            var_name,
            point_value,
            FiniteLimitSide::Left,
        )? == InfSign::Pos
            && finite_endpoint_argument_zero_tail_sign(
                ctx,
                expr,
                var_name,
                point_value,
                FiniteLimitSide::Right,
            )? == InfSign::Pos,
    )
}

pub(super) fn finite_endpoint_argument_zero_tail_negative_on_both_sides(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
) -> Option<bool> {
    Some(
        finite_endpoint_argument_zero_tail_sign(
            ctx,
            expr,
            var_name,
            point_value,
            FiniteLimitSide::Left,
        )? == InfSign::Neg
            && finite_endpoint_argument_zero_tail_sign(
                ctx,
                expr,
                var_name,
                point_value,
                FiniteLimitSide::Right,
            )? == InfSign::Neg,
    )
}

pub(super) fn finite_endpoint_unit_base_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    point_value: &BigRational,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    if let Ok(polynomial) = Polynomial::from_expr(ctx, expr, var_name) {
        let unit_gap = polynomial.sub(&Polynomial::one(var_name.to_string()));
        let (order, derivative) =
            finite_polynomial_local_order_and_derivative(&unit_gap, point_value)?;
        if order == 0 {
            return None;
        }
        return finite_local_tail_sign(&derivative, order, side);
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let denominator_value = denominator.eval(point_value);
    if denominator_value.is_zero() {
        return None;
    }

    let unit_gap_numerator = numerator.sub(&denominator);
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&unit_gap_numerator, point_value)?;
    if gap_order == 0 {
        return None;
    }

    let gap_tail = finite_local_tail_sign(&gap_derivative, gap_order, side)?;
    let denominator_tail = if denominator_value.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    Some(if gap_tail == denominator_tail {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

pub(super) fn multiply_tail_signs(lhs: InfSign, rhs: InfSign) -> InfSign {
    if lhs == rhs {
        InfSign::Pos
    } else {
        InfSign::Neg
    }
}

/// Functions `f` with `f(u) ~ u` as `u -> 0` (Taylor leading term exactly
/// `u`, i.e. `f(u)/u -> 1`). These are the first-order equivalent
/// infinitesimals. Cos/cosh are EXCLUDED: they tend to 1, not 0.
pub(super) fn is_first_order_zero_atom(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Sin
            | BuiltinFn::Tan
            | BuiltinFn::Asin
            | BuiltinFn::Arcsin
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Sinh
            | BuiltinFn::Tanh
    )
}

pub(super) fn is_finite_positive_domain_unary_builtin(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10 | BuiltinFn::Sqrt
    )
}

pub(super) fn finite_positive_domain_unary_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if !argument_value.is_positive() {
            return None;
        }
        if let Some(exact_result) =
            finite_positive_domain_unary_exact_numeric_result(ctx, builtin, &argument_value)
        {
            return Some(exact_result);
        }
        let value_expr = ctx.add(Expr::Number(argument_value));
        return Some(ctx.call_builtin(builtin, vec![value_expr]));
    }

    if let Some(exact_result) =
        finite_positive_domain_unary_exact_expr_result(ctx, builtin, argument_limit)
    {
        return Some(exact_result);
    }

    finite_expr_proven_positive(ctx, argument_limit)
        .then(|| ctx.call_builtin(builtin, vec![argument_limit]))
}

fn finite_positive_domain_unary_exact_numeric_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_value: &BigRational,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Sqrt => rational_sqrt(argument_value).map(|root| ctx.add(Expr::Number(root))),
        BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10 if argument_value.is_one() => {
            Some(ctx.num(0))
        }
        BuiltinFn::Log2 => {
            let base = BigRational::from_integer(BigInt::from(2));
            finite_exact_rational_log_result(ctx, &base, argument_value)
        }
        BuiltinFn::Log10 => {
            let base = BigRational::from_integer(BigInt::from(10));
            finite_exact_rational_log_result(ctx, &base, argument_value)
        }
        _ => None,
    }
}

fn finite_positive_domain_unary_exact_expr_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Ln => finite_ln_exact_expr_result(ctx, argument_limit),
        _ => None,
    }
}

pub(super) fn apply_finite_positive_domain_unary_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !is_finite_positive_domain_unary_builtin(builtin) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    finite_positive_domain_unary_result(ctx, builtin, argument_limit)
}

/// The lowest index with a nonzero coefficient, or None when all are zero.
pub(super) fn lowest_nonzero_order(coeffs: &[BigRational]) -> Option<usize> {
    coeffs.iter().position(|c| !c.is_zero())
}

/// Taylor coefficients of `expr` at 0 as a polynomial truncated to `order`,
/// for expressions built from polynomials, the standard analytic functions
/// (sin, cos, tan, exp, sinh, cosh, atan, asin, ln), and their sums,
/// products, integer powers, and compositions with a zero-at-0 argument.
/// Public Maclaurin expansion (expansion point `0`) of `expr` in `var_name`, truncated to
/// total degree `order` (inclusive), returned as an expression. `None` when the summand is
/// outside the supported analytic family — the same coverage the limit engine relies on
/// (polynomials, `exp`/`sin`/`cos`/`sinh`/`cosh`/`tan`/`atan`/`asin`/`ln`, their sums,
/// products, integer powers, and compositions with a series vanishing at 0). Expansion
/// points other than 0 are NOT handled here; the caller must restrict to `point = 0`.
pub fn taylor_series_at_zero_expr(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    order: usize,
) -> Option<ExprId> {
    let series = taylor_at_zero_with_rational(ctx, expr, var_name, order)?;
    Some(series.to_expr(ctx))
}

pub(super) fn taylor_at_zero(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    order: usize,
) -> Option<Polynomial> {
    if let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) {
        return Some(truncate_polynomial(&poly, order, var_name));
    }
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => Some(
            taylor_at_zero(ctx, lhs, var_name, order)?
                .add(&taylor_at_zero(ctx, rhs, var_name, order)?),
        ),
        Expr::Sub(lhs, rhs) => Some(
            taylor_at_zero(ctx, lhs, var_name, order)?
                .sub(&taylor_at_zero(ctx, rhs, var_name, order)?),
        ),
        Expr::Neg(inner) => {
            let minus_one = Polynomial::new(vec![-rational_one()], var_name.to_string());
            Some(taylor_at_zero(ctx, inner, var_name, order)?.mul(&minus_one))
        }
        Expr::Mul(lhs, rhs) => {
            let l = taylor_at_zero(ctx, lhs, var_name, order)?;
            let r = taylor_at_zero(ctx, rhs, var_name, order)?;
            Some(truncate_polynomial(&l.mul(&r), order, var_name))
        }
        Expr::Pow(base, exponent) => {
            // e^arg, or [series]^n for a non-negative integer n.
            if matches!(ctx.get(base), Expr::Constant(Constant::E)) {
                let inner = taylor_at_zero(ctx, exponent, var_name, order)?;
                return compose_standard_series(BuiltinFn::Exp, &inner, order, var_name);
            }
            let exp_value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if !exp_value.is_integer() || exp_value.is_negative() {
                return None;
            }
            let n: u32 = exp_value.to_integer().try_into().ok()?;
            let base_series = taylor_at_zero(ctx, base, var_name, order)?;
            let mut acc = Polynomial::one(var_name.to_string());
            for _ in 0..n {
                acc = truncate_polynomial(&acc.mul(&base_series), order, var_name);
            }
            Some(acc)
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let builtin = ctx.builtin_of(fn_id)?;
            let inner = taylor_at_zero(ctx, args[0], var_name, order)?;
            compose_standard_series(builtin, &inner, order, var_name)
        }
        _ => None,
    }
}

/// Horner evaluation of `sum_k coeffs[k] * inner^k` truncated to `order`,
/// valid when `inner(0) = 0` (so `inner^k` has order >= k).
pub(super) fn compose_with_zero_inner(
    coeffs: &[BigRational],
    inner: &Polynomial,
    order: usize,
    var_name: &str,
) -> Polynomial {
    let mut acc = Polynomial::new(vec![], var_name.to_string());
    for c in coeffs.iter().rev() {
        let c_poly = Polynomial::new(vec![c.clone()], var_name.to_string());
        acc = truncate_polynomial(&acc.mul(inner), order, var_name).add(&c_poly);
    }
    truncate_polynomial(&acc, order, var_name)
}

pub(super) fn neg_inf_sign(sign: InfSign) -> InfSign {
    match sign {
        InfSign::Pos => InfSign::Neg,
        InfSign::Neg => InfSign::Pos,
    }
}

pub(super) fn finite_limit_is_numeric_zero(ctx: &Context, expr: ExprId) -> bool {
    numeric_limit_value(ctx, expr).is_some_and(|value| value.is_zero())
}

pub(super) fn signed_abs_ratio_result(
    ctx: &mut Context,
    magnitude: BigRational,
    numerator_tail: InfSign,
    denominator_tail: InfSign,
) -> ExprId {
    let signed = if numerator_tail == denominator_tail {
        magnitude
    } else {
        -magnitude
    };
    ctx.add(Expr::Number(signed))
}

pub(super) fn limit_growth_sign(
    leading_coeff: &BigRational,
    degree: u32,
    approach: InfSign,
) -> InfSign {
    let coeff_positive = leading_coeff.is_positive();
    let power_positive = match approach {
        InfSign::Pos => true,
        InfSign::Neg => degree.is_multiple_of(2),
    };

    if coeff_positive == power_positive {
        InfSign::Pos
    } else {
        InfSign::Neg
    }
}

/// Tail sign of a general constant-base exponential `b^u` (`b` var-free and
/// provably positive with a provable position against 1): `b^u = e^(u·ln b)`,
/// so it follows `u`'s tail for `b > 1` and flips it for `0 < b < 1`. Keeps
/// the exp-tail contract — `Pos` means the atom diverges to `+inf`, `Neg`
/// means it decays to `0⁺` — so every consumer of the e-only classifiers
/// (dominance, log-of-exp, hierarchy) inherits provable bases like `π` or
/// `sqrt(2)` unchanged. Declines when the base-vs-1 comparison is unprovable
/// or the base is exactly 1.
pub(super) fn general_base_pow_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
    argument_tail: fn(&Context, ExprId, ExprId, InfSign) -> Option<InfSign>,
) -> Option<InfSign> {
    let Expr::Pow(base, exponent) = ctx.get(expr).clone() else {
        return None;
    };
    if depends_on(ctx, base, var) {
        return None;
    }
    let base_vs_one = constant_base_vs_one(ctx, base)?;
    let tail = argument_tail(ctx, exponent, var, approach)?;
    match base_vs_one {
        std::cmp::Ordering::Greater => Some(tail),
        std::cmp::Ordering::Less => Some(match tail {
            InfSign::Pos => InfSign::Neg,
            InfSign::Neg => InfSign::Pos,
        }),
        std::cmp::Ordering::Equal => None,
    }
}

pub(super) fn signed_pi_over_two(ctx: &mut Context, sign: InfSign) -> ExprId {
    let pi_over_two = TrigValue::PiDiv(2).to_expr(ctx);
    match sign {
        InfSign::Pos => pi_over_two,
        InfSign::Neg => ctx.add(Expr::Neg(pi_over_two)),
    }
}

pub(super) fn signed_unit_limit(ctx: &mut Context, sign: InfSign) -> ExprId {
    match sign {
        InfSign::Pos => ctx.num(1),
        InfSign::Neg => ctx.num(-1),
    }
}

/// Provable `base > 0` for a var-free constant base (exact rational compare,
/// or the exact interval bounds for `e`/`π` combinations).
pub(super) fn provably_positive_constant_base(ctx: &Context, base: ExprId) -> bool {
    use num_traits::{Signed, Zero};
    if let Some(value) = constant_rational_value(ctx, base) {
        return value.is_positive();
    }
    crate::root_forms::provable_const_minus_rational_sign(ctx, base, &BigRational::zero())
        == Some(std::cmp::Ordering::Greater)
}

/// F8b (Fase 3): PROVEN-zero multivariate limit by the polar bound (squeeze).
/// Decidable family, exact: `P/(x²+y²)^k` at the ORIGIN where EVERY monomial
/// of `P` has total degree `p > 2k` — then `|x^a·y^b| ≤ r^(a+b)` gives
/// `|f| ≤ (Σ|cᵢ|)·r^(p−2k) → 0`. Returns the bound display for the citation;
/// anything outside the exact family declines (`x·y/(x²+y²)` has p = 2k and
/// correctly falls to the DNE-by-paths driver — the two provers are
/// complementary by construction).
pub fn try_multivar_squeeze_zero(
    ctx: &mut Context,
    expr: ExprId,
    var_ids: &[ExprId],
    points: &[ExprId],
) -> Option<String> {
    use num_traits::Zero;
    if var_ids.len() != 2 || points.len() != 2 {
        return None;
    }
    for &pt in points {
        if !eval_rational_const_deep(ctx, pt).is_some_and(|v| v.is_zero()) {
            return None;
        }
    }
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let budget = crate::multipoly::PolyBudget::default();
    let num_poly = crate::multipoly::conversion::multipoly_from_expr(ctx, num, &budget).ok()?;
    let den_poly = crate::multipoly::conversion::multipoly_from_expr(ctx, den, &budget).ok()?;
    if num_poly.terms.is_empty() || den_poly.terms.is_empty() {
        return None;
    }
    // Denominator must be EXACTLY (x²+y²)^k: uniform total degree 2k and
    // equal to the constructed power (MultiPoly equality, same lex vars).
    let den_degrees: Vec<usize> = den_poly
        .terms
        .iter()
        .map(|(_, mono)| mono.iter().map(|&e| e as usize).sum())
        .collect();
    let d0 = *den_degrees.first()?;
    if d0 == 0 || d0 % 2 != 0 || den_degrees.iter().any(|&d| d != d0) {
        return None;
    }
    let k = d0 / 2;
    let (x_id, y_id) = (var_ids[0], var_ids[1]);
    let two = ctx.num(2);
    let x_sq = ctx.add(Expr::Pow(x_id, two));
    let two = ctx.num(2);
    let y_sq = ctx.add(Expr::Pow(y_id, two));
    let r_sq = ctx.add(Expr::Add(x_sq, y_sq));
    let k_expr = ctx.num(k as i64);
    let expected_expr = ctx.add(Expr::Pow(r_sq, k_expr));
    let expected =
        crate::multipoly::conversion::multipoly_from_expr(ctx, expected_expr, &budget).ok()?;
    if den_poly != expected {
        return None;
    }
    // Every numerator monomial strictly beats degree 2k.
    let min_deg = num_poly
        .terms
        .iter()
        .map(|(_, mono)| mono.iter().map(|&e| e as usize).sum::<usize>())
        .min()?;
    if min_deg <= 2 * k {
        return None;
    }
    Some(format!(
        "0 ≤ |f| ≤ C·r^{} con r² = x²+y² (cota polar: |x^a·y^b| ≤ r^(a+b)) — el límite es 0 por acotación",
        min_deg - 2 * k
    ))
}

pub(super) fn expr_is_zero(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::Zero;
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

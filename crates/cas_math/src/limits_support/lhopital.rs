//! `limits_support`: familia `lhopital`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

pub(super) fn apply_finite_one_sided_abs_polynomial_ratio_rule(
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
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (abs_scale, abs_arg) = scaled_abs_base(ctx, num)?;
    let abs_arg_poly = Polynomial::from_expr(ctx, abs_arg, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let (abs_order, abs_derivative) =
        finite_polynomial_local_order_and_derivative(&abs_arg_poly, &point_value)?;
    let (den_order, den_derivative) =
        finite_polynomial_local_order_and_derivative(&denominator, &point_value)?;
    if den_order == 0 {
        return None;
    }

    let numerator_tail = if abs_scale.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    let den_tail = finite_local_tail_sign(&den_derivative, den_order, side)?;

    if abs_order < den_order {
        return Some(signed_abs_ratio_infinity(ctx, numerator_tail, den_tail));
    }
    if abs_order > den_order {
        return Some(ctx.num(0));
    }

    let magnitude = abs_scale.abs() * abs_derivative.abs() / den_derivative.abs();
    Some(signed_abs_ratio_result(
        ctx,
        magnitude,
        numerator_tail,
        den_tail,
    ))
}

pub(super) fn apply_finite_bilateral_abs_polynomial_ratio_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left = apply_finite_one_sided_abs_polynomial_ratio_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Left,
    )?;
    let right = apply_finite_one_sided_abs_polynomial_ratio_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Right,
    )?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

/// `(c1 (a^g - 1)) / (c2 (b^h - 1))` at 0 -> the ratio of first-order
/// coefficients. Since `a^g - 1 ~ g ln a` for `g -> 0`, the numerator is
/// `~ c1 g'(0) ln a * x` and the denominator `~ c2 h'(0) ln b * x`, so the
/// limit is `(c1 g'(0) ln a) / (c2 h'(0) ln b)`. Resolves the classic
/// `(3^x - 1)/(2^x - 1) = ln 3 / ln 2`. Both sides must be a numeric-base
/// `a^x - 1` form vanishing at 0.
pub(super) fn apply_finite_general_exp_ratio_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    use num_traits::{One, Zero};
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    // (rational coefficient, base) of the first-order term scale*slope*ln(base).
    // The base may be rational OR a provable-positive constant != 1 (π, e,
    // e/π, sqrt(2), …): `a^g - 1 ~ g ln a` needs nothing more than `a > 0`,
    // and `a != 1` keeps both first-order coefficients nonzero.
    let first_order = |ctx: &mut Context, side: ExprId| -> Option<(BigRational, ExprId)> {
        let (scale, base, exponent) = scaled_general_power_zero_offset(ctx, side)?;
        if constant_base_vs_one(ctx, base)? == std::cmp::Ordering::Equal {
            return None;
        }
        let exp_poly = Polynomial::from_expr(ctx, exponent, &var_name).ok()?;
        // The exponent must vanish at 0 (so a^g -> 1) and have a linear term.
        if !exp_poly.eval(&BigRational::zero()).is_zero() {
            return None;
        }
        let slope = exp_poly.coeffs.get(1).cloned()?;
        if slope.is_zero() {
            return None;
        }
        Some((scale * slope, base))
    };
    let (num_coeff, num_base) = first_order(ctx, num)?;
    let (den_coeff, den_base) = first_order(ctx, den)?;
    // result = (num_coeff / den_coeff) * ln(num_base) / ln(den_base).
    let rational = &num_coeff / &den_coeff;
    // Equal bases: ln(a)/ln(a) = 1, so the limit is the bare rational ratio.
    if structurally_equal_expr(ctx, num_base, den_base) {
        return Some(ctx.add(Expr::Number(rational)));
    }
    let num_log = ctx.call_builtin(BuiltinFn::Ln, vec![num_base]);
    let den_log = ctx.call_builtin(BuiltinFn::Ln, vec![den_base]);
    let log_ratio = ctx.add(Expr::Div(num_log, den_log));
    if rational.is_one() {
        Some(log_ratio)
    } else {
        let rational_expr = ctx.add(Expr::Number(rational));
        Some(ctx.add(Expr::Mul(rational_expr, log_ratio)))
    }
}

/// `(c0 a^(g0) + c1 a^(g1) + ...) / h` at 0 where the numerator is a linear
/// combination of exponentials that vanishes at 0 and `h` is a polynomial
/// vanishing to first order: the limit is the ratio of first derivatives
/// `N'(0) / h'(0) = (sum c_i g_i'(0) ln a_i) / h'(0)`. Resolves the difference
/// of general-base exponentials `(a^x - b^x)/x -> ln(a) - ln(b)`, which the
/// single-power rule and the rational Taylor engine cannot reach (ln a is
/// transcendental).
/// `(c0 a^g0 + ...) / (d0 b^h0 + ...)` at 0 where BOTH sides are exponential
/// combinations vanishing at 0: the limit is the ratio of first derivatives
/// `N'(0)/D'(0)`. Resolves `(2^x - 3^x)/(5^x - 7^x) -> (ln 2 - ln 3)/(ln 5 -
/// ln 7)`, the two-sided sibling of (a^x-b^x)/x and (a^x-1)/(b^x-1).
pub(super) fn apply_finite_exp_combination_ratio_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if !crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let (num_log, num_const) = exp_combination_first_derivative(ctx, num, &var_name)?;
    let (den_log, den_const) = exp_combination_first_derivative(ctx, den, &var_name)?;
    // D'(0) must be PROVABLY nonzero for the first-order ratio to be the
    // limit: a zero-valued denominator combination (2^x + 2^(-x) - 2 is
    // quadratic at 0) would fabricate a division by zero. Exact decision —
    // a float epsilon here both missed true zeros under large coefficients
    // and declined true nonzeros.
    if exact_log_combination_is_zero(&den_log, &den_const) != Some(false) {
        return None;
    }
    // N'(0) = 0 over a nonzero D'(0): the numerator vanishes faster, limit 0.
    // EXACT zero only — a true-but-tiny N'(0) like ln(2) − 6931471805599453/
    // 10^16 must build the expression, never fold to 0 (P0 otherwise).
    if exact_log_combination_is_zero(&num_log, &num_const) == Some(true) {
        return Some(ctx.num(0));
    }
    let num_deriv = build_log_combination_expr(ctx, num_log, num_const);
    let den_deriv = build_log_combination_expr(ctx, den_log, den_const);
    Some(ctx.add(Expr::Div(num_deriv, den_deriv)))
}

/// L'Hôpital's rule for a genuine 0/0 quotient at a finite NON-ZERO point.
///
/// The point 0 is owned by the equivalent-infinitesimal and Maclaurin-Taylor
/// rules above (which also carry the educational small-angle narration), so this
/// rule deliberately declines there: its job is the gap they cannot reach, a 0/0
/// whose vanishing happens at a shifted point (`sin(x)/(x-pi)` at `pi`,
/// `tan(x)/sin(x)` at `pi`, `(1-cos(x-1))/(x-1)^2` at `1`). It differentiates the
/// numerator and denominator and re-evaluates the quotient's limit, repeating
/// while the form stays 0/0.
///
/// Soundness: L'Hôpital concludes `lim f/g = lim f'/g'` ONLY when the latter
/// exists. We therefore evaluate `lim f'` and `lim g'` through the full limit
/// machinery and act only on definite finite values: if either fails to resolve,
/// or the denominator's limit stays 0 while the numerator's does not (a pole), we
/// decline and the form remains an honest residual. The point-vanishing of both
/// parts is verified before differentiating, so non-0/0 quotients are left to the
/// ordinary substitution rules.
pub(super) fn apply_finite_lhopital_nonzero_point_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    // Point 0 is owned by the at-zero rules; only act on a shifted point.
    if crate::numeric_eval::as_rational_const(ctx, point).is_some_and(|p| p.is_zero()) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Div(num0, den0) = ctx.get(expr).clone() else {
        return None;
    };

    let depth = LHOPITAL_REENTRY_DEPTH.with(|d| d.get());
    if depth >= MAX_LHOPITAL_DEPTH {
        return None;
    }
    LHOPITAL_REENTRY_DEPTH.with(|d| d.set(depth + 1));
    let result = lhopital_evaluate(ctx, num0, den0, var, &var_name, point);
    LHOPITAL_REENTRY_DEPTH.with(|d| d.set(depth));
    result
}

fn lhopital_evaluate(
    ctx: &mut Context,
    mut num: ExprId,
    mut den: ExprId,
    var: ExprId,
    var_name: &str,
    point: ExprId,
) -> Option<ExprId> {
    for applied in 0..=MAX_LHOPITAL_DEPTH {
        // The differentiator emits unexpanded products (`2·(x-1)`); expand so the
        // polynomial/continuous rules recognize them when their limit is taken.
        // The differentiator emits unfolded exponents (`(x-1)^(2-1)`), which the
        // polynomial recognizer rejects; fold them so the limit of each part can
        // be taken by the ordinary rules.
        // The differentiator leaves arithmetic in exponents (`(x-1)^(2-1)`) that
        // the polynomial recognizer rejects; fold every constant subexpression to
        // a literal so the ordinary rules can take each part's limit.
        num = fold_constant_subexprs(ctx, num);
        den = fold_constant_subexprs(ctx, den);
        let limit_num = try_limit_rules_at_finite(ctx, num, var, point)?;
        let limit_den = try_limit_rules_at_finite(ctx, den, var, point)?;
        let num_value = limit_result_rational(ctx, limit_num)?;
        let den_value = limit_result_rational(ctx, limit_den)?;

        let num_zero = num_value.is_zero();
        let den_zero = den_value.is_zero();

        if den_zero && num_zero {
            // Genuine 0/0: differentiate both and apply L'Hôpital once more.
            if applied >= MAX_LHOPITAL_DEPTH {
                return None;
            }
            num = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
                ctx, num, var_name,
            )?;
            den = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
                ctx, den, var_name,
            )?;
            continue;
        }
        if den_zero {
            // Numerator does not vanish while the denominator does: a pole. Stay
            // an honest residual (the bilateral limit diverges or is signed-DNE).
            return None;
        }
        if applied == 0 {
            // Not a 0/0 to begin with: ordinary substitution owns this, not us.
            return None;
        }
        // Denominator's limit is nonzero after >=1 application: lim f/g = num/den.
        return Some(ctx.add(Expr::Number(num_value / den_value)));
    }
    None
}

//! `limits_support`: familia `support`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

/// Check if an expression depends on a specific variable id.
///
/// Uses iterative traversal to avoid recursion limits on deep trees.
pub(crate) fn depends_on(ctx: &Context, expr: ExprId, var: ExprId) -> bool {
    let mut stack = vec![expr];

    while let Some(current) = stack.pop() {
        if current == var {
            return true;
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => {}
            // A matrix depends on the variable iff any ENTRY does — skipping the
            // entries made `[[1/x,0],[0,1]]` read as var-free, so the constant
            // rule asserted the matrix itself as its own limit (P0, 2026-07-19).
            Expr::Matrix { data, .. } => {
                for entry in data {
                    stack.push(*entry);
                }
            }
            Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Parse a power expression with integer exponent.
///
/// Returns `(base, n)` if `expr` is `base^n` where `n` is an integer literal.
pub(crate) fn parse_pow_int(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let n = crate::expr_extract::extract_i64_integer(ctx, *exp)?;
            Some((*base, n))
        }
        _ => None,
    }
}

pub(super) fn finite_rational_polynomial_value(
    numerator: &Polynomial,
    denominator: &Polynomial,
    point: &BigRational,
) -> Option<BigRational> {
    let mut numerator = numerator.clone();
    let mut denominator = denominator.clone();
    let max_derivative_steps = numerator.degree().max(denominator.degree()) + 1;

    for _ in 0..=max_derivative_steps {
        let numerator_value = numerator.eval(point);
        let denominator_value = denominator.eval(point);
        if !denominator_value.is_zero() {
            return Some(numerator_value / denominator_value);
        }
        if !numerator_value.is_zero() || (numerator.is_zero() && denominator.is_zero()) {
            return None;
        }

        numerator = numerator.derivative();
        denominator = denominator.derivative();
    }

    None
}

pub(super) fn finite_polynomial_local_order_and_derivative(
    polynomial: &Polynomial,
    point: &BigRational,
) -> Option<(usize, BigRational)> {
    let mut current = polynomial.clone();
    for order in 0..=polynomial.degree() {
        let value = current.eval(point);
        if !value.is_zero() {
            return Some((order, value));
        }
        current = current.derivative();
    }
    None
}

pub(super) fn finite_local_tail_sign(
    derivative_value: &BigRational,
    order: usize,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    if derivative_value.is_zero() {
        return None;
    }
    let positive = derivative_value.is_positive()
        == (side == FiniteLimitSide::Right || order.is_multiple_of(2));
    Some(if positive { InfSign::Pos } else { InfSign::Neg })
}

pub(super) fn matching_finite_bilateral_one_sided_result(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    match (
        infinity_sign_of_expr(ctx, left),
        infinity_sign_of_expr(ctx, right),
    ) {
        (Some(left_sign), Some(right_sign)) if left_sign == right_sign => return Some(left),
        _ => {}
    }

    match (ctx.get(left), ctx.get(right)) {
        (Expr::Number(left_value), Expr::Number(right_value)) if left_value == right_value => {
            Some(left)
        }
        _ if left == right => Some(left),
        _ => None,
    }
}

pub(super) fn structurally_equal_expr(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    lhs == rhs || cas_ast::ordering::compare_expr(ctx, lhs, rhs) == std::cmp::Ordering::Equal
}

pub(super) fn try_limit_rules_at_finite(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if let Some(result) = apply_static_empty_real_domain_rule(ctx, expr, var) {
        return Some(result);
    }
    if let Some(result) = apply_constant_rule(ctx, expr, var) {
        return Some(result);
    }
    if expr == var && !depends_on(ctx, point, var) {
        return Some(point);
    }
    if let Some(result) = apply_finite_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_rational_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_radical_difference_conjugate_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_radical_conjugate_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_bilateral_rational_polynomial_pole_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_abs_polynomial_ratio_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_sign_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_trig_power_pole_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_trig_ratio_power_pole_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_sine_zero_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_exp_zero_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_general_exp_zero_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_general_exp_ratio_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_exp_combination_ratio_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_log_unit_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_equivalent_infinitesimal_quotient_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_taylor_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_exp_linear_combination_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_log_endpoint_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_elementary_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_acosh_polynomial_endpoint_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_inverse_trig_polynomial_endpoint_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_sqrt_endpoint_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_square_root_power_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_cube_root_power_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_integer_power_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_positive_domain_unary_composition_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_partial_domain_unary_composition_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) =
        apply_finite_domain_checked_trig_unary_composition_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_binary_log_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_total_real_unary_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_squeeze_bounded_product_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_one_to_infinity_power_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_bilateral_even_saturating_pole_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_lhopital_nonzero_point_quotient_rule(ctx, expr, var, point) {
        return Some(result);
    }

    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            // Fast path: operand-wise when both sides have a limit. Computed
            // without `?` so a None does not bail before the combine
            // fallback (mirrors the Sub arm below).
            if let (Some(lhs_limit), Some(rhs_limit)) = (
                try_limit_rules_at_finite(ctx, lhs, var, point),
                try_limit_rules_at_finite(ctx, rhs, var, point),
            ) {
                if let Some(result) = finite_add_result_checked(ctx, lhs_limit, rhs_limit) {
                    return Some(result);
                }
            }
            // Fallback: combine `lhs + rhs` over a common denominator and
            // retry the single fraction — `x/(x-1) + 1/(1-x)` = `1` even
            // though each operand diverges at x = 1. The combined form is a
            // `Div`, so this does not loop.
            let combined = combine_sum_over_common_denominator(ctx, lhs, rhs)?;
            try_limit_rules_at_finite(ctx, combined, var, point)
        }
        Expr::Sub(lhs, rhs) => {
            // Fast path: operand-wise limits when BOTH converge (or are
            // determinate infinities). Computed without `?` so a None on
            // either side does NOT bail before the combine fallback below.
            if let (Some(lhs_limit), Some(rhs_limit)) = (
                try_limit_rules_at_finite(ctx, lhs, var, point),
                try_limit_rules_at_finite(ctx, rhs, var, point),
            ) {
                if let Some(result) = finite_sub_result(ctx, lhs_limit, rhs_limit) {
                    return Some(result);
                }
            }
            // Fallback: combine `lhs - rhs` over a common denominator and
            // retry the limit of the single fraction. Reached both when the
            // operands are the indeterminate same-sign ∞ - ∞ (`1/sin²x -
            // 1/x² -> 1/3`) AND when an operand has no rule-level limit at
            // all — `1/sin(x)` at 0 is bilaterally undefined so the operand
            // split yields None, yet `(x - sin x)/(x sin x) -> 0`. The
            // combined form is a `Div` (not a `Sub`), so this does not loop.
            let combined = combine_difference_over_common_denominator(ctx, lhs, rhs)?;
            try_limit_rules_at_finite(ctx, combined, var, point)
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_finite(ctx, lhs, var, point)?;
            let rhs_limit = try_limit_rules_at_finite(ctx, rhs, var, point)?;
            finite_mul_result(ctx, lhs_limit, rhs_limit)
        }
        Expr::Div(num, den) => {
            let num_limit = try_limit_rules_at_finite(ctx, num, var, point)?;
            let den_limit = try_limit_rules_at_finite(ctx, den, var, point)?;
            finite_div_result(ctx, num_limit, den_limit)
        }
        Expr::Neg(inner) => {
            let inner_limit = try_limit_rules_at_finite(ctx, inner, var, point)?;
            Some(finite_neg_result(ctx, inner_limit))
        }
        _ => None,
    }
}

pub(super) fn limit_value_infinite_sign(ctx: &Context, value: ExprId) -> Option<i32> {
    match ctx.get(value) {
        Expr::Constant(Constant::Infinity) => Some(1),
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            Some(-1)
        }
        _ => None,
    }
}

pub(super) fn numeric_limit_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Neg(inner) => numeric_limit_value(ctx, *inner).map(|value| -value),
        Expr::Add(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? + numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Sub(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? - numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Mul(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? * numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Div(num, den) => {
            let den_value = numeric_limit_value(ctx, *den)?;
            if den_value.is_zero() {
                return None;
            }
            Some(numeric_limit_value(ctx, *num)? / den_value)
        }
        _ => None,
    }
}

pub(super) fn finite_expr_proven_positive(ctx: &Context, expr: ExprId) -> bool {
    crate::prove_sign::prove_positive_depth_with(ctx, expr, 4, true, |_, _, _| {
        crate::tri_proof::TriProof::Unknown
    })
    .is_proven()
}

pub(super) fn scaled_square_root_base(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    if let Some(radicand) = extract_square_root_base(ctx, expr) {
        return Some((BigRational::from_integer(BigInt::from(1)), radicand));
    }

    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                if scale.is_zero() {
                    return None;
                }
                return extract_square_root_base(ctx, rhs).map(|radicand| (scale, radicand));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                if scale.is_zero() {
                    return None;
                }
                return extract_square_root_base(ctx, lhs).map(|radicand| (scale, radicand));
            }
            None
        }
        Expr::Neg(inner) => {
            scaled_square_root_base(ctx, inner).map(|(scale, radicand)| (-scale, radicand))
        }
        _ => None,
    }
}

pub(super) fn signed_abs_ratio_infinity(
    ctx: &mut Context,
    numerator_tail: InfSign,
    denominator_tail: InfSign,
) -> ExprId {
    let sign = if numerator_tail == denominator_tail {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    mk_infinity(ctx, sign)
}

pub(super) fn rational_one() -> BigRational {
    BigRational::from_integer(BigInt::from(1))
}

pub(super) fn constant_rational_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr).clone() {
        Expr::Number(value) => Some(value),
        Expr::Neg(inner) => Some(-constant_rational_value(ctx, inner)?),
        Expr::Div(num, den) => {
            let den_value = constant_rational_value(ctx, den)?;
            if den_value.is_zero() {
                return None;
            }
            Some(constant_rational_value(ctx, num)? / den_value)
        }
        _ => None,
    }
}

/// Exact comparison of a var-free positive base against 1: rational bases
/// compare exactly; constant expressions over `e`/`π` fall to the exact
/// rational interval bounds of the surd-sign chokepoint (`e/π < 1` is a
/// PROOF, never a float estimate). `None` when positivity or the comparison
/// is not provable — callers must decline, so a wrong growth class is never
/// fabricated.
pub(super) fn constant_base_vs_one(ctx: &Context, base: ExprId) -> Option<std::cmp::Ordering> {
    use num_traits::{One, Signed, Zero};
    use std::cmp::Ordering;
    if let Some(value) = constant_rational_value(ctx, base) {
        if !value.is_positive() {
            return None;
        }
        return Some(value.cmp(&BigRational::one()));
    }
    if crate::root_forms::provable_const_minus_rational_sign(ctx, base, &BigRational::zero())?
        != Ordering::Greater
    {
        return None;
    }
    crate::root_forms::provable_const_minus_rational_sign(ctx, base, &BigRational::one())
}

pub(super) fn expr_is_one(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::One;
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

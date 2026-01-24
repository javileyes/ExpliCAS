//! Limit rules for V1: x → ±∞.
//!
//! Rules are applied in order:
//! 1. ConstantRule - expression independent of variable
//! 2. VariableRule - expression is the variable itself  
//! 3. PowerRule - x^n with integer n
//! 4. RationalPolyRule - P(x)/Q(x) polynomial division

use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;

use crate::rules::infinity::{mk_infinity, InfSign};
use crate::Budget;

use super::helpers::{depends_on, limit_sign, mk_inf, parse_pow_int};
use super::types::Approach;

/// Rule 1: Constant - lim c = c (if c doesn't depend on var)
pub fn apply_constant_rule(ctx: &Context, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if !depends_on(ctx, expr, var) {
        Some(expr)
    } else {
        None
    }
}

/// Rule 2: Variable - lim x = ±∞ based on approach
pub fn apply_variable_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
) -> Option<ExprId> {
    if expr != var {
        return None;
    }

    Some(match approach {
        Approach::PosInfinity => mk_infinity(ctx, InfSign::Pos),
        Approach::NegInfinity => mk_infinity(ctx, InfSign::Neg),
    })
}

/// Rule 3: Power - lim x^n for integer n
///
/// - n > 0: ±∞ (sign depends on approach and parity)
/// - n = 0: 1
/// - n < 0: 0
pub fn apply_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
) -> Option<ExprId> {
    let (base, n) = parse_pow_int(ctx, expr)?;

    // Base must be exactly the variable
    if base != var {
        return None;
    }

    if n == 0 {
        return Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1)))));
    }

    if n < 0 {
        return Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))));
    }

    // n > 0: infinity with appropriate sign
    let sign = limit_sign(approach, n);
    Some(mk_inf(ctx, sign))
}

/// Rule 4: Simple fraction 1/x^n - special case of rational
///
/// lim 1/x^n = 0 for n > 0
pub fn apply_reciprocal_power_rule(ctx: &mut Context, expr: ExprId, var: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Numerator must be constant (not depend on var)
    if depends_on(ctx, num, var) {
        return None;
    }

    // Denominator must be x^n with n > 0, or just x
    let power = if den == var {
        1
    } else if let Some((base, n)) = parse_pow_int(ctx, den) {
        if base != var || n <= 0 {
            return None;
        }
        n
    } else {
        return None;
    };

    if power > 0 {
        Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))))
    } else {
        None
    }
}

/// Rule 5: Rational polynomial P(x)/Q(x)
///
/// Compares degrees of numerator and denominator:
/// - deg(P) < deg(Q) → 0
/// - deg(P) = deg(Q) → lc(P)/lc(Q) (if both are numeric constants)
/// - deg(P) > deg(Q) → ±∞ (sign depends on leading coeffs and approach)
///
/// Returns `None` if:
/// - Expression is not a fraction
/// - Cannot convert to polynomial (contains trig, roots, etc.)
/// - Leading coefficients depend on other variables
pub fn try_rational_poly_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
) -> Option<ExprId> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    // Match Div(num, den)
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Get variable name
    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    // Conservative budget for polynomial conversion
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    // Convert numerator and denominator to polynomials
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Get variable index in polynomial
    // If var not in poly, it's constant wrt var (degree 0)
    let var_idx_num = p_num.var_index(&var_name);
    let var_idx_den = p_den.var_index(&var_name);

    // If neither contains the variable, constant rule handles it
    if var_idx_num.is_none() && var_idx_den.is_none() {
        return None; // Let constant rule handle it
    }

    // Check for zero denominator polynomial
    if p_den.is_zero() {
        return None; // Division by zero - don't handle here
    }

    // Get degrees
    let deg_p = var_idx_num.map(|idx| p_num.degree_in(idx)).unwrap_or(0);
    let deg_q = var_idx_den.map(|idx| p_den.degree_in(idx)).unwrap_or(0);

    // Get leading coefficients
    let lc_p = var_idx_num
        .map(|idx| p_num.leading_coeff_in(idx))
        .unwrap_or_else(|| p_num.clone());
    let lc_q = var_idx_den
        .map(|idx| p_den.leading_coeff_in(idx))
        .unwrap_or_else(|| p_den.clone());

    // Both leading coefficients must be numeric constants
    let lc_p_val = lc_p.constant_value()?;
    let lc_q_val = lc_q.constant_value()?;

    // Case 1: deg(P) < deg(Q) → 0
    if deg_p < deg_q {
        return Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))));
    }

    // Case 2: deg(P) = deg(Q) → lc(P)/lc(Q)
    if deg_p == deg_q {
        let ratio = &lc_p_val / &lc_q_val;
        return Some(ctx.add(Expr::Number(ratio)));
    }

    // Case 3: deg(P) > deg(Q) → ±∞
    // Sign = sign(lc_p/lc_q) * sign(x^k) where k = deg_p - deg_q
    let k = deg_p - deg_q;
    let ratio = &lc_p_val / &lc_q_val;

    // Determine sign of ratio
    use num_traits::Signed;
    let ratio_positive = ratio.is_positive();

    // Determine sign of x^k for the approach
    let xk_positive = match approach {
        Approach::PosInfinity => true,
        Approach::NegInfinity => k % 2 == 0, // x^even is positive, x^odd is negative
    };

    // Combined sign: positive if both same, negative if different
    let result_positive = ratio_positive == xk_positive;
    let sign = if result_positive {
        InfSign::Pos
    } else {
        InfSign::Neg
    };

    Some(mk_inf(ctx, sign))
}

/// Try all limit rules in order.
///
/// Returns Some(result) if a rule applies, None if no rule applies.
pub fn try_limit_rules(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
    _budget: &mut Budget,
) -> Option<ExprId> {
    // Rule 1: Constant
    if let Some(r) = apply_constant_rule(ctx, expr, var) {
        return Some(r);
    }

    // Rule 2: Variable
    if let Some(r) = apply_variable_rule(ctx, expr, var, approach) {
        return Some(r);
    }

    // Rule 3: Power
    if let Some(r) = apply_power_rule(ctx, expr, var, approach) {
        return Some(r);
    }

    // Rule 3b: Reciprocal power (1/x^n)
    if let Some(r) = apply_reciprocal_power_rule(ctx, expr, var) {
        return Some(r);
    }

    // Rule 4: Rational polynomial P(x)/Q(x)
    if let Some(r) = try_rational_poly_rule(ctx, expr, var, approach) {
        return Some(r);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Constant;
    use cas_parser::parse;
    use num_traits::Zero;

    fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
        parse(s, ctx).expect("parse failed")
    }

    // Test 1: lim_{x→∞} 5 = 5
    #[test]
    fn test_limit_constant() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5");
        let x = parse_expr(&mut ctx, "x");

        let result = apply_constant_rule(&ctx, expr, x);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), expr);
    }

    // Test 3: lim_{x→∞} x = ∞
    #[test]
    fn test_limit_variable_pos_inf() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let result = apply_variable_rule(&mut ctx, x, x, Approach::PosInfinity);
        assert!(result.is_some());
        assert!(matches!(
            ctx.get(result.unwrap()),
            Expr::Constant(Constant::Infinity)
        ));
    }

    // Test 4: lim_{x→-∞} x = -∞
    #[test]
    fn test_limit_variable_neg_inf() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let result = apply_variable_rule(&mut ctx, x, x, Approach::NegInfinity);
        assert!(result.is_some());

        let r = result.unwrap();
        assert!(matches!(ctx.get(r), Expr::Neg(_)));
    }

    // Test 5: lim_{x→∞} x^2 = ∞
    #[test]
    fn test_limit_x_squared_pos_inf() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^2");
        let x = parse_expr(&mut ctx, "x");

        let result = apply_power_rule(&mut ctx, expr, x, Approach::PosInfinity);
        assert!(result.is_some());
        assert!(matches!(
            ctx.get(result.unwrap()),
            Expr::Constant(Constant::Infinity)
        ));
    }

    // Test 6: lim_{x→-∞} x^3 = -∞
    #[test]
    fn test_limit_x_cubed_neg_inf() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^3");
        let x = parse_expr(&mut ctx, "x");

        let result = apply_power_rule(&mut ctx, expr, x, Approach::NegInfinity);
        assert!(result.is_some());

        let r = result.unwrap();
        assert!(matches!(ctx.get(r), Expr::Neg(_)));
    }

    // Test 1: lim_{x→∞} 1/x = 0
    #[test]
    fn test_limit_one_over_x() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "1/x");
        let x = parse_expr(&mut ctx, "x");

        let result = apply_reciprocal_power_rule(&mut ctx, expr, x);
        assert!(result.is_some());

        if let Expr::Number(n) = ctx.get(result.unwrap()) {
            assert!(n.is_zero());
        } else {
            panic!("Expected Number(0)");
        }
    }

    // Test 2: lim_{x→∞} 5/x^3 = 0
    #[test]
    fn test_limit_five_over_x_cubed() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5/x^3");
        let x = parse_expr(&mut ctx, "x");

        let result = apply_reciprocal_power_rule(&mut ctx, expr, x);
        assert!(result.is_some());

        if let Expr::Number(n) = ctx.get(result.unwrap()) {
            assert!(n.is_zero());
        } else {
            panic!("Expected Number(0)");
        }
    }

    // ========== RATIONAL POLYNOMIAL RULE TESTS (V1.1) ==========

    // Test 7: lim_{x→∞} (x^2+1)/(2*x^2-3) = 1/2
    #[test]
    fn test_rational_poly_equal_degree() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "(x^2+1)/(2*x^2-3)");
        let x = parse_expr(&mut ctx, "x");

        let result = try_rational_poly_rule(&mut ctx, expr, x, Approach::PosInfinity);
        assert!(result.is_some(), "Should resolve (x²+1)/(2x²-3)");

        if let Expr::Number(n) = ctx.get(result.unwrap()) {
            // 1/2
            assert_eq!(*n, BigRational::new(BigInt::from(1), BigInt::from(2)));
        } else {
            panic!("Expected Number(1/2)");
        }
    }

    // Test 8: lim_{x→∞} (3*x^3+1)/(x^3-7) = 3
    #[test]
    fn test_rational_poly_equal_degree_three() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "(3*x^3+1)/(x^3-7)");
        let x = parse_expr(&mut ctx, "x");

        let result = try_rational_poly_rule(&mut ctx, expr, x, Approach::PosInfinity);
        assert!(result.is_some());

        if let Expr::Number(n) = ctx.get(result.unwrap()) {
            assert_eq!(*n, BigRational::from_integer(BigInt::from(3)));
        } else {
            panic!("Expected Number(3)");
        }
    }

    // Test 9: lim_{x→∞} x^3/x^2 = ∞
    #[test]
    fn test_rational_poly_higher_num_degree() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^3/x^2");
        let x = parse_expr(&mut ctx, "x");

        let result = try_rational_poly_rule(&mut ctx, expr, x, Approach::PosInfinity);
        assert!(result.is_some());
        assert!(matches!(
            ctx.get(result.unwrap()),
            Expr::Constant(Constant::Infinity)
        ));
    }

    // Test 10: lim_{x→∞} x^2/x^3 = 0
    #[test]
    fn test_rational_poly_lower_num_degree() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^2/x^3");
        let x = parse_expr(&mut ctx, "x");

        let result = try_rational_poly_rule(&mut ctx, expr, x, Approach::PosInfinity);
        assert!(result.is_some());

        if let Expr::Number(n) = ctx.get(result.unwrap()) {
            assert!(n.is_zero());
        } else {
            panic!("Expected Number(0)");
        }
    }

    // Test 11: lim_{x→-∞} x^3/x^2 = -∞ (odd degree difference)
    #[test]
    fn test_rational_poly_neg_inf_odd_k() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^3/x^2");
        let x = parse_expr(&mut ctx, "x");

        let result = try_rational_poly_rule(&mut ctx, expr, x, Approach::NegInfinity);
        assert!(result.is_some());

        // Should be Neg(Infinity)
        assert!(matches!(ctx.get(result.unwrap()), Expr::Neg(_)));
    }

    // Test 12: lim_{x→-∞} x^4/x^3 = -∞ (k=1, odd, positive ratio)
    #[test]
    fn test_rational_poly_neg_inf_x4_x3() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^4/x^3");
        let x = parse_expr(&mut ctx, "x");

        let result = try_rational_poly_rule(&mut ctx, expr, x, Approach::NegInfinity);
        assert!(result.is_some());

        // k=1 (odd), ratio=1 (positive), x^1 at -∞ is negative
        // Result: positive * negative = negative → -∞
        assert!(matches!(ctx.get(result.unwrap()), Expr::Neg(_)));
    }

    // Test 13: sin(x) → None (not a polynomial)
    #[test]
    fn test_rational_poly_rejects_sin() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "sin(x)/x");
        let x = parse_expr(&mut ctx, "x");

        let result = try_rational_poly_rule(&mut ctx, expr, x, Approach::PosInfinity);
        assert!(result.is_none(), "Should not handle sin(x)/x");
    }

    // Test 14: (y*x^2)/x^2 → None (leading coeff depends on y)
    #[test]
    fn test_rational_poly_rejects_param_coeff() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "(y*x^2)/x^2");
        let x = parse_expr(&mut ctx, "x");

        let result = try_rational_poly_rule(&mut ctx, expr, x, Approach::PosInfinity);
        assert!(
            result.is_none(),
            "Should reject when leading coeff is not numeric"
        );
    }
}

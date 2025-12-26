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

    // TODO: Rule 4: RationalPolyRule - requires polynomial infrastructure

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
}

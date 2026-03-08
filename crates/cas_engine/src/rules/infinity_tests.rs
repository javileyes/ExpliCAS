use super::infinity::*;
use cas_ast::{Constant, Context, Expr, ExprId};
use cas_math::infinity_support::{classify_finiteness, Finiteness, InfSign};
use cas_parser::parse;
use num_traits::Zero;

fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
    parse(s, ctx).expect("parse failed")
}

// ========== REGRESSION TESTS ==========

/// Test 118: Infinity addition absorption
/// 1000 + infinity → infinity
#[test]
fn test_118_infinity_addition() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "1000 + infinity");

    let result = add_infinity_absorption(&mut ctx, expr);
    assert!(result.is_some(), "Test 118: Should absorb 1000 + ∞ → ∞");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
        "Test 118: Result should be infinity"
    );
}

/// Test 119: Division by infinity
/// 5 / infinity → 0
#[test]
fn test_119_infinity_division() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "5 / infinity");

    let result = div_by_infinity(&mut ctx, expr);
    assert!(result.is_some(), "Test 119: Should simplify 5/∞ → 0");

    let new_expr = result.unwrap().new_expr;
    if let Expr::Number(n) = ctx.get(new_expr) {
        assert!(n.is_zero(), "Test 119: Result should be 0");
    } else {
        panic!("Test 119: Result should be Number(0)");
    }
}

/// Test: Infinity plus negative infinity is indeterminate
/// infinity + -infinity → undefined
#[test]
fn test_infinity_plus_neg_infinity_indeterminate() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "infinity + -infinity");

    let result = add_infinity_absorption(&mut ctx, expr);
    assert!(result.is_some(), "Should detect ∞ + (-∞) indeterminate");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
        "Result should be Undefined"
    );
}

/// Test: Infinity minus infinity is indeterminate
/// infinity - infinity → undefined
#[test]
fn test_infinity_minus_infinity_indeterminate() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "infinity - infinity");

    let result = add_infinity_absorption(&mut ctx, expr);
    assert!(result.is_some(), "Should detect ∞ - ∞ indeterminate");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
        "Result should be Undefined"
    );
}

/// Test: Zero times infinity is indeterminate
/// 0 * infinity → undefined
#[test]
fn test_zero_times_infinity_indeterminate() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "0 * infinity");

    let result = mul_zero_infinity(&mut ctx, expr);
    assert!(result.is_some(), "Should detect 0·∞ indeterminate");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
        "Result should be Undefined"
    );
}

/// Test: Infinity times zero (commutative)
/// infinity * 0 → undefined
#[test]
fn test_infinity_times_zero_indeterminate() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "infinity * 0");

    let result = mul_zero_infinity(&mut ctx, expr);
    assert!(result.is_some(), "Should detect ∞·0 indeterminate");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
        "Result should be Undefined"
    );
}

/// Test: Addition with negative infinity
/// 1000 + -infinity → -infinity
#[test]
fn test_addition_with_negative_infinity() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "1000 + -infinity");

    let result = add_infinity_absorption(&mut ctx, expr);
    assert!(result.is_some(), "Should absorb 1000 + (-∞) → -∞");

    let new_expr = result.unwrap().new_expr;
    // Should be Neg(Infinity)
    if let Expr::Neg(inner) = ctx.get(new_expr) {
        assert!(
            matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)),
            "Result should be -infinity"
        );
    } else {
        panic!("Result should be Neg(Infinity)");
    }
}

/// Test: Division by negative infinity also gives 0
/// 5 / -infinity → 0
#[test]
fn test_division_by_negative_infinity() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "5 / -infinity");

    let result = div_by_infinity(&mut ctx, expr);
    assert!(result.is_some(), "Should simplify 5/(-∞) → 0");

    let new_expr = result.unwrap().new_expr;
    if let Expr::Number(n) = ctx.get(new_expr) {
        assert!(n.is_zero(), "Result should be 0");
    } else {
        panic!("Result should be Number(0)");
    }
}

/// Test: Conservative policy - variable + infinity should NOT simplify
#[test]
fn test_conservative_variable_plus_infinity() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x + infinity");

    let result = add_infinity_absorption(&mut ctx, expr);
    assert!(
        result.is_none(),
        "Conservative: x + ∞ should NOT simplify (x could be -∞)"
    );
}

// ========== EXTENDED RULES TESTS ==========

/// Test: Finite times infinity
/// 3 * infinity → infinity
#[test]
fn test_finite_times_infinity() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "3 * infinity");

    let result = mul_finite_infinity(&mut ctx, expr);
    assert!(result.is_some(), "Should simplify 3 * ∞ → ∞");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
        "Result should be infinity"
    );
}

/// Test: Negative finite times infinity
/// (-2) * infinity → -infinity
#[test]
fn test_negative_finite_times_infinity() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "(-2) * infinity");

    let result = mul_finite_infinity(&mut ctx, expr);
    assert!(result.is_some(), "Should simplify (-2) * ∞ → -∞");

    let new_expr = result.unwrap().new_expr;
    // Should be Neg(Infinity)
    if let Expr::Neg(inner) = ctx.get(new_expr) {
        assert!(
            matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)),
            "Result should be -infinity"
        );
    } else {
        panic!("Result should be Neg(Infinity)");
    }
}

/// Test: Infinity times finite (commutative)
/// infinity * 5 → infinity
#[test]
fn test_infinity_times_finite() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "infinity * 5");

    let result = mul_finite_infinity(&mut ctx, expr);
    assert!(result.is_some(), "Should simplify ∞ * 5 → ∞");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
        "Result should be infinity"
    );
}

/// Test: Infinity divided by finite
/// infinity / 2 → infinity
#[test]
fn test_infinity_div_finite() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "infinity / 2");

    let result = inf_div_finite(&mut ctx, expr);
    assert!(result.is_some(), "Should simplify ∞ / 2 → ∞");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
        "Result should be infinity"
    );
}

/// Test: Infinity divided by negative finite
/// infinity / (-3) → -infinity
#[test]
fn test_infinity_div_negative_finite() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "infinity / (-3)");

    let result = inf_div_finite(&mut ctx, expr);
    assert!(result.is_some(), "Should simplify ∞ / (-3) → -∞");

    let new_expr = result.unwrap().new_expr;
    // Should be Neg(Infinity)
    if let Expr::Neg(inner) = ctx.get(new_expr) {
        assert!(
            matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)),
            "Result should be -infinity"
        );
    } else {
        panic!("Result should be Neg(Infinity)");
    }
}

/// Test: Negative infinity divided by negative finite
/// -infinity / (-2) → infinity (negative * negative = positive)
#[test]
fn test_neg_infinity_div_neg_finite() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "-infinity / (-2)");

    let result = inf_div_finite(&mut ctx, expr);
    assert!(result.is_some(), "Should simplify -∞ / (-2) → ∞");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
        "Result should be positive infinity"
    );
}

/// Test: Symmetric - (-infinity) - (-infinity) is indeterminate
#[test]
fn test_neg_inf_minus_neg_inf_indeterminate() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "-infinity - (-infinity)");

    let result = add_infinity_absorption(&mut ctx, expr);
    assert!(result.is_some(), "Should detect -∞ - (-∞) indeterminate");

    let new_expr = result.unwrap().new_expr;
    assert!(
        matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
        "Result should be Undefined"
    );
}

/// Test: Conservative policy - variable * infinity should NOT simplify
#[test]
fn test_conservative_variable_times_infinity() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x * infinity");

    let result = mul_finite_infinity(&mut ctx, expr);
    assert!(result.is_none(), "Conservative: x * ∞ should NOT simplify");
}

// ========== CLASSIFY_FINITENESS TESTS ==========

/// Test: classify_finiteness for numbers
#[test]
fn test_classify_number_is_finite() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "42");

    assert_eq!(
        classify_finiteness(&ctx, expr),
        Finiteness::FiniteLiteral,
        "Numbers should be FiniteLiteral"
    );
}

/// Test: classify_finiteness for pi
#[test]
fn test_classify_pi_is_finite() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "pi");

    assert_eq!(
        classify_finiteness(&ctx, expr),
        Finiteness::FiniteLiteral,
        "Pi should be FiniteLiteral"
    );
}

/// Test: classify_finiteness for infinity
#[test]
fn test_classify_infinity() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "infinity");

    assert_eq!(
        classify_finiteness(&ctx, expr),
        Finiteness::Infinity(InfSign::Pos),
        "infinity should be Infinity(Pos)"
    );
}

/// Test: classify_finiteness for negative infinity
#[test]
fn test_classify_neg_infinity() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "-infinity");

    assert_eq!(
        classify_finiteness(&ctx, expr),
        Finiteness::Infinity(InfSign::Neg),
        "-infinity should be Infinity(Neg)"
    );
}

/// Test: classify_finiteness for variables is Unknown
#[test]
fn test_classify_variable_is_unknown() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x");

    assert_eq!(
        classify_finiteness(&ctx, expr),
        Finiteness::Unknown,
        "Variables should be Unknown"
    );
}

/// Test: classify_finiteness for undefined is Unknown
#[test]
fn test_classify_undefined_is_unknown() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "undefined");

    assert_eq!(
        classify_finiteness(&ctx, expr),
        Finiteness::Unknown,
        "Undefined should be Unknown (not a known value)"
    );
}

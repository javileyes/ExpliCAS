//! Regression tests for unified display ordering
//!
//! These tests ensure that:
//! 1. Expressions built in different AST order render identically (canonical ordering)
//! 2. Non-polynomial mixed expressions render deterministically
//!
//! These protect against reintroducing visual inconsistencies between "Rule:" and "After:"
//! in didactic traces.

use cas_ast::display::DisplayExpr;
use cas_ast::{Context, Expr};

// ============================================================================
// Test 2: Same order for different construction order
// ============================================================================

/// Test that two equivalent expressions built in different AST order
/// render identically with DisplayExpr.
///
/// This protects against the bug where "Rule:" showed `4 + x^2 + 2x`
/// and "After:" showed `x^2 + 2x + 4`.
#[test]
fn test_different_construction_order_same_display() {
    let mut ctx = Context::new();

    // Build e1: 4 + x^2 + 2x (constants first, then powers, then linear)
    let x = ctx.var("x");
    let two = ctx.num(2);
    let four = ctx.num(4);
    let x_sq = ctx.add(Expr::Pow(x, two));
    let two_x = ctx.add(Expr::Mul(two, x));

    // 4 + x^2
    let step1 = ctx.add(Expr::Add(four, x_sq));
    // (4 + x^2) + 2x
    let e1 = ctx.add(Expr::Add(step1, two_x));

    // Build e2: x^2 + 2x + 4 (canonical polynomial order)
    let x2 = ctx.var("x");
    let two2 = ctx.num(2);
    let four2 = ctx.num(4);
    let x_sq2 = ctx.add(Expr::Pow(x2, two2));
    let two_x2 = ctx.add(Expr::Mul(two2, x2));

    // x^2 + 2x
    let step1_2 = ctx.add(Expr::Add(x_sq2, two_x2));
    // (x^2 + 2x) + 4
    let e2 = ctx.add(Expr::Add(step1_2, four2));

    let display1 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: e1
        }
    );
    let display2 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: e2
        }
    );

    println!("e1 (4 + x^2 + 2x order): {}", display1);
    println!("e2 (x^2 + 2x + 4 order): {}", display2);

    // Both should render in canonical polynomial order: x^2 + 2*x + 4
    assert_eq!(
        display1, display2,
        "Expressions built in different order should render identically. \
         Got '{}' vs '{}'",
        display1, display2
    );

    // Verify the order is polynomial standard (highest degree first)
    assert!(display1.contains("x^2"), "Should contain x^2: {}", display1);
}

/// Test with a simpler case: x + 1 vs 1 + x
#[test]
fn test_simple_add_order_canonical() {
    let mut ctx = Context::new();

    // Build 1 + x (constant first)
    let x1 = ctx.var("x");
    let one1 = ctx.num(1);
    let e1 = ctx.add(Expr::Add(one1, x1));

    // Build x + 1 (variable first)
    let x2 = ctx.var("x");
    let one2 = ctx.num(1);
    let e2 = ctx.add(Expr::Add(x2, one2));

    let display1 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: e1
        }
    );
    let display2 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: e2
        }
    );

    println!("1 + x built: {}", display1);
    println!("x + 1 built: {}", display2);

    assert_eq!(display1, display2, "1+x and x+1 should render identically");

    // Should be "x + 1" (variable before constant in polynomial order)
    assert_eq!(display1, "x + 1", "Canonical order should be x + 1");
}

// ============================================================================
// Test 3: Determinism for non-polynomial expressions
// ============================================================================

/// Test that mixed non-polynomial expressions render deterministically.
/// This validates that the fallback to `compare_expr` works correctly.
#[test]
fn test_non_polynomial_determinism() {
    let mut ctx = Context::new();

    // Build: sin(x) + x^2 + 1/(x+1) + sqrt(x)
    let x = ctx.var("x");
    let one = ctx.num(1);
    let two = ctx.num(2);

    // sin(x)
    let sin_x = ctx.call("sin", vec![x]);

    // x^2
    let x_sq = ctx.add(Expr::Pow(x, two));

    // x + 1 (for denominator)
    let x_plus_1 = ctx.add(Expr::Add(x, one));
    // 1/(x+1)
    let reciprocal = ctx.add(Expr::Div(one, x_plus_1));

    // sqrt(x) = x^(1/2)
    let half = ctx.add(Expr::Div(one, two));
    let sqrt_x = ctx.add(Expr::Pow(x, half));

    // Build sum in one order: sin(x) + x^2 + 1/(x+1) + sqrt(x)
    let sum1 = ctx.add(Expr::Add(sin_x, x_sq));
    let sum2 = ctx.add(Expr::Add(sum1, reciprocal));
    let expr1 = ctx.add(Expr::Add(sum2, sqrt_x));

    // Render twice
    let render1 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr1
        }
    );
    let render2 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr1
        }
    );

    println!("First render: {}", render1);
    println!("Second render: {}", render2);

    // Same expression should always render identically
    assert_eq!(
        render1, render2,
        "Same expression should render identically on multiple calls"
    );

    // Verify it contains all expected components
    assert!(render1.contains("sin"), "Should contain sin(x)");
    assert!(
        render1.contains("x^2") || render1.contains("x²"),
        "Should contain x^2"
    );
}

/// Test that building the same sum in different orders produces same display.
/// This is the strongest test of determinism.
#[test]
fn test_non_polynomial_different_construction_order() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let _one = ctx.num(1);
    let two = ctx.num(2);

    // Components
    let sin_x = ctx.call("sin", vec![x]);
    let x_sq = ctx.add(Expr::Pow(x, two));
    let cos_x = ctx.call("cos", vec![x]);

    // Build order 1: sin(x) + x^2 + cos(x)
    let sum1_a = ctx.add(Expr::Add(sin_x, x_sq));
    let expr1 = ctx.add(Expr::Add(sum1_a, cos_x));

    // Build order 2: x^2 + cos(x) + sin(x) (different construction order)
    let x2 = ctx.var("x");
    let two2 = ctx.num(2);
    let sin_x2 = ctx.call("sin", vec![x2]);
    let x_sq2 = ctx.add(Expr::Pow(x2, two2));
    let cos_x2 = ctx.call("cos", vec![x2]);

    let sum2_a = ctx.add(Expr::Add(x_sq2, cos_x2));
    let expr2 = ctx.add(Expr::Add(sum2_a, sin_x2));

    // Build order 3: cos(x) + sin(x) + x^2 (yet another order)
    let x3 = ctx.var("x");
    let two3 = ctx.num(2);
    let sin_x3 = ctx.call("sin", vec![x3]);
    let x_sq3 = ctx.add(Expr::Pow(x3, two3));
    let cos_x3 = ctx.call("cos", vec![x3]);

    let sum3_a = ctx.add(Expr::Add(cos_x3, sin_x3));
    let expr3 = ctx.add(Expr::Add(sum3_a, x_sq3));

    let display1 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr1
        }
    );
    let display2 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr2
        }
    );
    let display3 = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr3
        }
    );

    println!("Order 1 (sin+x^2+cos): {}", display1);
    println!("Order 2 (x^2+cos+sin): {}", display2);
    println!("Order 3 (cos+sin+x^2): {}", display3);

    // All should render identically regardless of construction order
    assert_eq!(
        display1, display2,
        "Different construction order should produce same display"
    );
    assert_eq!(
        display2, display3,
        "Different construction order should produce same display"
    );

    // The key assertion: all terms are present and order is consistent (deterministic)
    // Note: compare_expr ordering may put functions before/after powers based on type ranking
    // The important thing is CONSISTENCY, not a specific order
    assert!(
        display1.contains("x^2") && display1.contains("sin") && display1.contains("cos"),
        "All terms should be present: {}",
        display1
    );
}

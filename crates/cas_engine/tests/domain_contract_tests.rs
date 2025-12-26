//! Domain mode contract tests.
//!
//! These tests verify the behavior of factor cancellation under different
//! domain modes (Strict, Assume, Generic).
//!
//! # Contract
//!
//! - **Generic** (default): Allow `x/x → 1` (legacy CAS behavior)
//! - **Strict**: Only cancel if factor is provably non-zero
//! - **Assume**: Like Strict, but uses user-provided assumptions (future)
//!
//! # Tests organized by mode:
//!
//! - `test_generic_*`: Current behavior (should pass now)
//! - `test_strict_*`: Expected behavior after gate (will fail until implemented)

use cas_ast::Context;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper: simplify with default options (Generic mode)
fn simplify_generic(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");
    let (result, _) = simplifier.simplify(expr);
    format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

/// Helper: simplify with Strict domain mode
fn simplify_strict(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    // Use Strict domain mode
    let opts = cas_engine::SimplifyOptions {
        domain: cas_engine::DomainMode::Strict,
        ..Default::default()
    };

    let (result, _) = simplifier.simplify_with_options(expr, opts);
    format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// =============================================================================
// Generic Mode Tests (should pass now - preserves existing behavior)
// =============================================================================

#[test]
fn test_generic_x_div_x_simplifies_to_1() {
    // Generic mode: x/x → 1 (classic CAS behavior)
    let result = simplify_generic("x/x");
    assert_eq!(result, "1", "Generic mode should simplify x/x to 1");
}

#[test]
fn test_generic_2_div_2_simplifies_to_1() {
    // Both modes: 2/2 → 1 (2 is provably non-zero)
    let result = simplify_generic("2/2");
    assert_eq!(result, "1");
}

#[test]
fn test_generic_pi_div_pi_simplifies_to_1() {
    // Both modes: π/π → 1 (π is provably non-zero)
    let result = simplify_generic("pi/pi");
    assert_eq!(result, "1");
}

// =============================================================================
// Strict Mode Tests (will fail until gate is implemented)
// =============================================================================

#[test]
#[ignore = "pending: gate CancelCommonFactorsRule for domain mode"]
fn test_strict_x_div_x_stays_unchanged() {
    // Strict mode: x/x should NOT simplify (x could be 0)
    let result = simplify_strict("x/x");
    assert_eq!(result, "x/x", "Strict mode should NOT simplify x/x");
}

#[test]
fn test_strict_2_div_2_simplifies_to_1() {
    // Strict mode: 2/2 → 1 (2 is provably non-zero)
    let result = simplify_strict("2/2");
    assert_eq!(result, "1", "Strict mode should simplify 2/2 to 1");
}

#[test]
fn test_strict_neg3_div_neg3_simplifies_to_1() {
    // Strict mode: (-3)/(-3) → 1 (-3 is provably non-zero)
    let result = simplify_strict("(-3)/(-3)");
    assert_eq!(result, "1", "Strict mode should simplify (-3)/(-3) to 1");
}

#[test]
fn test_strict_pi_div_pi_simplifies_to_1() {
    // Strict mode: π/π → 1 (π is provably non-zero)
    let result = simplify_strict("pi/pi");
    assert_eq!(result, "1", "Strict mode should simplify pi/pi to 1");
}

#[test]
#[ignore = "pending: gate CancelCommonFactorsRule for domain mode"]
fn test_strict_2x_div_2x_stays_unchanged() {
    // Strict mode: (2*x)/(2*x) should NOT simplify to 1 (x could be 0)
    // It might partially simplify but shouldn't collapse to 1
    let result = simplify_strict("(2*x)/(2*x)");
    // The result should either be unchanged OR be x/x (partial cancel of 2)
    // It should NOT be "1"
    assert_ne!(
        result, "1",
        "Strict mode should NOT fully simplify (2*x)/(2*x) to 1"
    );
}

#[test]
#[ignore = "pending: gate CancelCommonFactorsRule for domain mode"]
fn test_strict_x_squared_div_x_squared_stays_unchanged() {
    // Strict mode: x²/x² should NOT simplify (x could be 0)
    let result = simplify_strict("x^2/x^2");
    assert_ne!(result, "1", "Strict mode should NOT simplify x^2/x^2 to 1");
}

// =============================================================================
// Proof Helper Tests
// =============================================================================

#[test]
fn test_prove_nonzero_numbers() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let zero = ctx.num(0);
    let two = ctx.num(2);
    let neg_three = ctx.num(-3);
    let half = ctx.add(cas_ast::Expr::Number(num_rational::BigRational::new(
        1.into(),
        2.into(),
    )));

    assert_eq!(prove_nonzero(&ctx, zero), Proof::Disproven);
    assert_eq!(prove_nonzero(&ctx, two), Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, neg_three), Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, half), Proof::Proven);
}

#[test]
fn test_prove_nonzero_constants() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let pi = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Pi));
    let e = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));
    let i = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::I));

    assert_eq!(prove_nonzero(&ctx, pi), Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, e), Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, i), Proof::Proven);
}

#[test]
fn test_prove_nonzero_variables() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let x = ctx.var("x");
    let y = ctx.var("y");

    assert_eq!(prove_nonzero(&ctx, x), Proof::Unknown);
    assert_eq!(prove_nonzero(&ctx, y), Proof::Unknown);
}

#[test]
fn test_prove_nonzero_products() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let two = ctx.num(2);
    let three = ctx.num(3);
    let zero = ctx.num(0);
    let x = ctx.var("x");

    // 2 * 3 = 6, proven nonzero
    let two_times_three = ctx.add(cas_ast::Expr::Mul(two, three));
    assert_eq!(prove_nonzero(&ctx, two_times_three), Proof::Proven);

    // 2 * 0 = 0, disproven nonzero
    let two_times_zero = ctx.add(cas_ast::Expr::Mul(two, zero));
    assert_eq!(prove_nonzero(&ctx, two_times_zero), Proof::Disproven);

    // 2 * x = unknown (x could be 0)
    let two_times_x = ctx.add(cas_ast::Expr::Mul(two, x));
    assert_eq!(prove_nonzero(&ctx, two_times_x), Proof::Unknown);
}

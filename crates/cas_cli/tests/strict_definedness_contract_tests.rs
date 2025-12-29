//! Contract tests for DomainMode::Strict definedness preservation.
//!
//! # Contract: Strict Definedness Invariant
//!
//! "No rewrite shall reduce the set of points where an expression may be undefined,
//! unless definedness is proven."
//!
//! These tests verify that definedness-erasing rewrites only fire when:
//! - In Strict: provably safe (literals, algebraic identities)
//! - In Assume: with AssumptionEvent
//! - In Generic: silently

use cas_ast::display::DisplayExpr;
use cas_engine::Simplifier;

/// Helper: simplify with Strict domain mode
fn simplify_strict(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");

    let opts = cas_engine::SimplifyOptions {
        domain: cas_engine::DomainMode::Strict,
        ..Default::default()
    };

    let (result, _) = simplifier.simplify_with_options(expr, opts);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

/// Helper: simplify with Assume domain mode
fn simplify_assume(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");

    let opts = cas_engine::SimplifyOptions {
        domain: cas_engine::DomainMode::Assume,
        ..Default::default()
    };

    let (result, _) = simplifier.simplify_with_options(expr, opts);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

/// Helper: simplify with Generic domain mode
fn simplify_generic(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");
    let (result, _) = simplifier.simplify(expr);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// =============================================================================
// Contract 1: Zero Numerator with Symbolic Denominator
// =============================================================================

#[test]
fn strict_zero_over_symbolic_does_not_collapse() {
    // 0/(x+1) should NOT collapse to 0 in Strict (x+1 could be 0)
    let result = simplify_strict("0/(x+1)");
    assert!(
        result.contains("0") && result.contains("/"),
        "Strict should preserve 0/(x+1), got: {}",
        result
    );
}

#[test]
fn generic_zero_over_symbolic_collapses() {
    // 0/(x+1) SHOULD collapse to 0 in Generic
    let result = simplify_generic("0/(x+1)");
    assert_eq!(
        result.trim(),
        "0",
        "Generic should collapse 0/(x+1) to 0, got: {}",
        result
    );
}

#[test]
fn assume_zero_over_symbolic_collapses() {
    // 0/(x+1) SHOULD collapse to 0 in Assume (with assumption)
    let result = simplify_assume("0/(x+1)");
    assert_eq!(
        result.trim(),
        "0",
        "Assume should collapse 0/(x+1) to 0, got: {}",
        result
    );
}

// =============================================================================
// Contract 2: Additive Inverse with Undefined Risk
// =============================================================================

#[test]
fn strict_subtraction_with_undefined_does_not_collapse() {
    // x/(x+1) - x/(x+1) should NOT collapse to 0 in Strict
    let result = simplify_strict("x/(x+1) - x/(x+1)");
    // Should preserve the subtraction form (not be just "0")
    assert!(
        result.contains("-") || result.contains("/"),
        "Strict should preserve x/(x+1) - x/(x+1), got: {}",
        result
    );
}

#[test]
fn generic_subtraction_with_undefined_collapses() {
    // x/(x+1) - x/(x+1) SHOULD collapse to 0 in Generic
    let result = simplify_generic("x/(x+1) - x/(x+1)");
    assert_eq!(
        result.trim(),
        "0",
        "Generic should collapse to 0, got: {}",
        result
    );
}

// =============================================================================
// Contract 3: Zero Numerator with Proven Denominator
// =============================================================================

#[test]
fn strict_zero_over_literal_collapses() {
    // 0/2 SHOULD collapse to 0 in Strict (2 is provably nonzero)
    let result = simplify_strict("0/2");
    assert_eq!(
        result.trim(),
        "0",
        "Strict should collapse 0/2 to 0, got: {}",
        result
    );
}

#[test]
fn strict_zero_over_proven_nonzero_collapses() {
    // 0/(1+1) SHOULD collapse (1+1 = 2, provably nonzero)
    let result = simplify_strict("0/(1+1)");
    assert_eq!(
        result.trim(),
        "0",
        "Strict should collapse 0/(1+1) to 0, got: {}",
        result
    );
}

// =============================================================================
// Contract 4: Zero Annihilation with Undefined Risk
// =============================================================================

/// MulZeroRule is now gated by has_undefined_risk.
/// This test verifies the fix is working correctly.
#[test]
fn strict_zero_times_undefined_does_not_collapse() {
    // 0 * (x/(x+1)) should NOT collapse to 0 in Strict
    let result = simplify_strict("0 * (x/(x+1))");
    // Should preserve multiplication (not be just "0")
    assert!(
        result.contains("*")
            || result.contains("/")
            || result.contains("0 ·")
            || result.contains("0·"),
        "Strict should preserve 0 * (x/(x+1)), got: {}",
        result
    );
}

#[test]
fn generic_zero_times_undefined_collapses() {
    // 0 * (x/(x+1)) SHOULD collapse to 0 in Generic
    let result = simplify_generic("0 * (x/(x+1))");
    assert_eq!(
        result.trim(),
        "0",
        "Generic should collapse to 0, got: {}",
        result
    );
}

// =============================================================================
// Contract 5: Simple Division Cancellation
// =============================================================================

#[test]
fn strict_x_over_x_does_not_collapse() {
    // x/x should NOT collapse to 1 in Strict (x could be 0)
    let result = simplify_strict("x/x");
    assert!(
        result.contains("x") && result.contains("/"),
        "Strict should preserve x/x, got: {}",
        result
    );
}

#[test]
fn generic_x_over_x_collapses() {
    // x/x SHOULD collapse to 1 in Generic
    let result = simplify_generic("x/x");
    assert_eq!(
        result.trim(),
        "1",
        "Generic should collapse x/x to 1, got: {}",
        result
    );
}

#[test]
fn assume_x_over_x_collapses() {
    // x/x SHOULD collapse to 1 in Assume (with assumption)
    let result = simplify_assume("x/x");
    assert_eq!(
        result.trim(),
        "1",
        "Assume should collapse x/x to 1, got: {}",
        result
    );
}

// =============================================================================
// Bonus: Zero Exponent with Symbolic Base
// =============================================================================

#[test]
fn strict_x_pow_zero_does_not_collapse() {
    // x^0 should NOT collapse to 1 in Strict (0^0 is undefined)
    let result = simplify_strict("x^0");
    assert!(
        result.contains("^") || result.contains("x"),
        "Strict should preserve x^0, got: {}",
        result
    );
}

#[test]
fn generic_x_pow_zero_collapses() {
    // x^0 SHOULD collapse to 1 in Generic
    let result = simplify_generic("x^0");
    assert_eq!(
        result.trim(),
        "1",
        "Generic should collapse x^0 to 1, got: {}",
        result
    );
}

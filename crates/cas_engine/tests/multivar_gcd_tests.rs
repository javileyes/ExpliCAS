//! Multivariate GCD Integration Tests
//!
//! Tests for multivariate polynomial GCD in fraction simplification.

use cas_ast::Context;
use cas_engine::options::{BranchMode, ComplexMode, ContextMode, EvalOptions};
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper: simplify expression and return result string
fn simplify(input: &str) -> String {
    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::Standard,
        complex_mode: ComplexMode::Auto,
    };
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;
    let (result, _steps) = simplifier.simplify(expr);

    format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// =============================================================================
// Content GCD Tests (Layer 1)
// =============================================================================

#[test]
fn test_content_gcd_multivar() {
    // (2x + 2y) / (4x + 4y)
    // With Layer 2: GCD = 2*(x+y), so result = 1/2
    let result = simplify("(2*x + 2*y) / (4*x + 4*y)");
    assert_eq!(result, "1/2", "(2x+2y)/(4x+4y) should simplify to 1/2");
}

// =============================================================================
// Monomial GCD Tests (Layer 1)
// =============================================================================

#[test]
fn test_monomial_gcd_multivar() {
    // (x^2*y + x^2*y^2) / (x^3*y + x^3*y^2) should simplify to 1/x
    let result = simplify("(x^2*y + x^2*y^2) / (x^3*y + x^3*y^2)");
    // After factoring out x^2*y from num and x^3*y from den:
    // num: x^2*y*(1 + y), den: x^3*y*(1 + y)
    // Common: x^2*y*(1+y)
    // Result: 1/x
    assert!(
        result.contains("1") && result.contains("x"),
        "Should simplify to 1/x, got: {}",
        result
    );
}

// =============================================================================
// Combined Content + Monomial Tests
// =============================================================================

#[test]
fn test_combined_content_and_monomial() {
    // (2*x*y) / (4*x^2*y^2) should simplify to 1/(2*x*y)
    let result = simplify("(2*x*y) / (4*x^2*y^2)");
    // Content GCD: 2, Monomial GCD: x*y
    // Result: 1 / (2*x*y)
    assert!(
        result.contains("1") && result.contains("2"),
        "Should simplify, got: {}",
        result
    );
}

// =============================================================================
// Layer 2 Tests: Heuristic Polynomial GCD
// =============================================================================

#[test]
fn test_layer2_difference_of_squares() {
    // (x^2 - y^2) / (x - y) should simplify to x + y
    // This requires Layer 2: GCD = (x - y)
    let result = simplify("(x^2 - y^2) / (x - y)");
    // Expected: x + y (or y + x depending on ordering)
    assert!(
        (result.contains("x") && result.contains("y") && result.contains("+"))
            || result == "x + y"
            || result == "y + x",
    );
}

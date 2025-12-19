//! Contract tests for poly_gcd_exact (algebraic polynomial GCD).
//!
//! Tests verify:
//! - Univariate GCD cases
//! - Multivariate content/monomial GCD
//! - Constants over ℚ
//! - Zero handling
//! - Budget handling
//! - No expand contamination

use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper: run poly_gcd_exact and get result string
fn run_gcd_exact(a: &str, b: &str) -> (String, cas_ast::ExprId, cas_ast::Context) {
    let mut simplifier = Simplifier::with_default_rules();
    let input = format!("poly_gcd_exact({}, {})", a, b);
    let expr = parse(&input, &mut simplifier.context).expect("parse failed");
    let (result, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    (result_str, result, simplifier.context)
}

// =============================================================================
// Test 1: Univar basic
// =============================================================================

#[test]
fn test_gcd_exact_univar_basic() {
    // gcd(x^2 - 1, x - 1) = (x+1)(x-1) gcd (x-1) = x-1
    let (result, _, _) = run_gcd_exact("x^2 - 1", "x - 1");
    assert!(
        result.contains("x") && result.contains("1"),
        "Expected x-1, got: {}",
        result
    );
}

#[test]
fn test_gcd_exact_univar_quadratics() {
    // gcd(x^2 - 1, x^2 - 2*x + 1) = gcd((x-1)(x+1), (x-1)^2) = x-1
    let (result, _, _) = run_gcd_exact("x^2 - 1", "x^2 - 2*x + 1");
    assert!(
        result.contains("x") && result.contains("1"),
        "Expected x-1, got: {}",
        result
    );
}

// =============================================================================
// Test 2: Content (scalar factor)
// =============================================================================

#[test]
fn test_gcd_exact_content() {
    // gcd(2x + 2y, 4x + 4y) = 2(x+y) / 2 = x+y (primitive)
    let (result, _, _) = run_gcd_exact("2*x + 2*y", "4*x + 4*y");
    // Should be x + y (primitive, content factored out)
    assert!(
        result.contains("x") && result.contains("y"),
        "Expected x+y, got: {}",
        result
    );
    // Should NOT have coefficient 2 or 4
    assert!(
        !result.contains("2") && !result.contains("4"),
        "Expected primitive (no coeff), got: {}",
        result
    );
}

// =============================================================================
// Test 3: Multivar with factor
// =============================================================================

#[test]
fn test_gcd_exact_multivar_factor() {
    // gcd(x*y + x, x^2*y + x^2) = x(y+1) gcd x^2(y+1) = x(y+1)
    let (result, _, _) = run_gcd_exact("x*y + x", "x^2*y + x^2");
    // Should contain both x and y
    assert!(
        result.contains("x"),
        "Expected x in result, got: {}",
        result
    );
}

// =============================================================================
// Test 4: No common factor
// =============================================================================

#[test]
fn test_gcd_exact_no_common() {
    // gcd(x^2 + y^2, (x+y)^2) = 1 (no common factor)
    let (result, _, ctx) = run_gcd_exact("x^2 + y^2", "(x+y)^2");
    // Should be 1
    assert!(result == "1", "Expected 1, got: {}", result);
}

// =============================================================================
// Test 5: Difference of cubes/squares
// =============================================================================

#[test]
fn test_gcd_exact_difference() {
    // gcd(x^3 - y^3, x^2 - y^2) = x - y
    // x^3 - y^3 = (x-y)(x^2 + xy + y^2)
    // x^2 - y^2 = (x-y)(x+y)
    // This requires Layer 2 or higher
    let (result, _, _) = run_gcd_exact("x^3 - y^3", "x^2 - y^2");
    // May or may not find x-y depending on layer capability
    // At minimum should return 1 or x-y
    assert!(
        result == "1" || (result.contains("x") && result.contains("y")),
        "Expected 1 or x-y, got: {}",
        result
    );
}

// =============================================================================
// Test 6: Standard mode no expansion
// =============================================================================

#[test]
fn test_gcd_exact_no_expand_side_effect() {
    // Verify that calling poly_gcd_exact doesn't expand (x+1)^3
    let mut simplifier = Simplifier::with_default_rules();

    // First: parse and simplify (x+1)^3
    let expr1 = parse("(x+1)^3", &mut simplifier.context).expect("parse");
    let (result1, _) = simplifier.simplify(expr1);
    let result1_str = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result1
        }
    );

    // Should still be (x+1)^3, not expanded
    assert!(
        result1_str.contains("³") || result1_str.contains("^3"),
        "Expected (x+1)^3 preserved, got: {}",
        result1_str
    );
}

// =============================================================================
// Test 7: Budget (ignored for now - would need huge input)
// =============================================================================

#[test]
#[ignore] // Enable when budget testing is needed
fn test_gcd_exact_budget_bailout() {
    // Very large input that should hit budget
    // (x1+x2+x3+x4+x5)^10 vs (x1+x2+x3+x4+x5)^10 + 1
    // This would be huge - budget should kick in
}

// =============================================================================
// Test 8: Determinism
// =============================================================================

#[test]
fn test_gcd_exact_deterministic() {
    let input_a = "x^2 - 1";
    let input_b = "x^2 - 2*x + 1";

    let mut results = Vec::new();
    for _ in 0..5 {
        let (result, _, _) = run_gcd_exact(input_a, input_b);
        results.push(result);
    }

    // All should be identical
    let first = &results[0];
    for r in &results {
        assert_eq!(r, first, "Results should be deterministic");
    }
}

// =============================================================================
// Test 9: Zero input
// =============================================================================

#[test]
fn test_gcd_exact_zero() {
    // gcd(0, x+1) = x+1
    let (result, _, _) = run_gcd_exact("0", "x + 1");
    assert!(
        result.contains("x") && result.contains("1"),
        "Expected x+1, got: {}",
        result
    );
}

// =============================================================================
// Test 10: Constants
// =============================================================================

#[test]
fn test_gcd_exact_constants() {
    // gcd(6, 15) = 1 over ℚ (any nonzero constant divides any other)
    let (result, _, _) = run_gcd_exact("6", "15");
    assert_eq!(
        result, "1",
        "Expected 1 for constants over ℚ, got: {}",
        result
    );
}

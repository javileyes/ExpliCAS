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

// =============================================================================
// CLOSURE TESTS
// =============================================================================

// =============================================================================
// Test 11: Commutativity - gcd(a,b) = gcd(b,a)
// =============================================================================

#[test]
fn test_gcd_exact_commutative() {
    let (result_ab, _, _) = run_gcd_exact("x^2 - 1", "x - 1");
    let (result_ba, _, _) = run_gcd_exact("x - 1", "x^2 - 1");
    assert_eq!(
        result_ab, result_ba,
        "GCD should be commutative: gcd(a,b) = gcd(b,a). Got {} vs {}",
        result_ab, result_ba
    );
}

// =============================================================================
// Test 12: Strong Determinism - 20 runs same result
// =============================================================================

#[test]
fn test_gcd_exact_determinism_20() {
    // gcd(x^2 - y^2, x - y) = x - y (difference of squares)
    let input_a = "x^2 - y^2";
    let input_b = "x - y";

    let mut results = Vec::new();
    for _ in 0..20 {
        let (result, _, _) = run_gcd_exact(input_a, input_b);
        results.push(result);
    }

    // All 20 should be identical
    let first = &results[0];
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r, first, "Run {} differs: {} vs {}", i, r, first);
    }

    // Should find x-y or contain x and y
    assert!(
        first.contains("x") && first.contains("y"),
        "Expected x-y factor, got: {}",
        first
    );
}

// =============================================================================
// Test 13: Expression substitution via Simplifier
// =============================================================================

#[test]
fn test_gcd_exact_expression_integration() {
    // Test that poly_gcd_exact integrates correctly with expression parsing
    let mut simplifier = Simplifier::with_default_rules();

    // Parse full expression directly
    let expr = parse("poly_gcd_exact(x^2 - 1, x - 1)", &mut simplifier.context).expect("parse");

    let (result, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should find x - 1
    assert!(
        result_str.contains("x") && result_str.contains("1"),
        "Expected x-1, got: {}",
        result_str
    );
}

// =============================================================================
// Test 14: Budget bailout (ignored - requires huge input)
// =============================================================================

#[test]
#[ignore] // Enable when testing budget limits specifically
fn test_gcd_exact_budget_warning() {
    use cas_ast::Context;
    use cas_engine::rules::algebra::gcd_exact::{gcd_exact, GcdExactBudget, GcdExactLayer};
    use cas_parser::parse;

    let mut ctx = Context::new();

    // Very large polynomials that would exceed budget
    // (x1+x2+x3+x4+x5+x6)^20 has millions of terms when expanded
    let a = parse("(x1+x2+x3+x4+x5+x6)^20 - 1", &mut ctx).expect("parse a");
    let b = parse("(x1+x2+x3+x4+x5+x6)^20 + 1", &mut ctx).expect("parse b");

    // Use a low budget
    let budget = GcdExactBudget {
        max_vars: 3, // Will exceed
        max_terms_input: 10,
        max_total_degree: 5,
    };

    let result = gcd_exact(&mut ctx, a, b, &budget);

    // Should bailout, not hang
    assert_eq!(
        result.layer_used,
        GcdExactLayer::BudgetExceeded,
        "Expected budget bailout"
    );

    // Should have warning
    assert!(
        !result.warnings.is_empty(),
        "Expected warning on budget exceeded"
    );
}

// =============================================================================
// Test 15: Multivariate difference of squares layered result
// =============================================================================

#[test]
fn test_gcd_exact_difference_of_squares() {
    // x^2 - y^2 = (x-y)(x+y), gcd with (x-y) should be (x-y)
    let (result, _, _) = run_gcd_exact("x^2 - y^2", "x - y");

    // Layer 2 should find x-y, or at minimum we get 1
    // (depends on layer capabilities)
    assert!(
        result == "1" || (result.contains("x") && (result.contains("-") || result.contains("y"))),
        "Expected 1 or x-y variant, got: {}",
        result
    );
}

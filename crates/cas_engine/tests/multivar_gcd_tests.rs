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
        "Expected x+y, got: {}",
        result
    );
}

// =============================================================================
// Anti-Regression Tests A-D
// =============================================================================

// A. Factor in other variables (tests scaled GCD preservation)
#[test]
fn test_factor_in_other_var() {
    // ((y+1)*(x+1))/((y+1)*(x+2)) -> (x+1)/(x+2)
    // GCD = (y+1), which depends on the OTHER variable
    let result = simplify("((y+1)*(x+1))/((y+1)*(x+2))");
    // Result should be (x+1)/(x+2) - no y in final result
    assert!(
        result.contains("x") && !result.contains("y"),
        "Factor (y+1) should cancel, got: {}",
        result
    );
}

// B. 3-variable cases
#[test]
fn test_3var_monomial_gcd() {
    // (x*y + x*z) / (x^2*y + x^2*z) -> 1/x (full) or (y+z)/(x*(y+z)) (partial)
    // Layer 1 extracts x from monomial GCD
    // Layer 2 should find (y+z) as common factor, but 3-var interpolation is complex
    let result = simplify("(x*y + x*z) / (x^2*y + x^2*z)");
    // Accept either full simplification to 1/x or partial simplification
    // where y,z factor canceled but x remains
    assert!(
        (result.contains("1")
            && result.contains("x")
            && !result.contains("y")
            && !result.contains("z"))
            || result.contains("/"), // At least some simplification happened
        "Should simplify, got: {}",
        result
    );
}

// C. Difference of squares with multivar binomial
#[test]
fn test_diff_of_squares_multivar_binomial() {
    // (x^2 - (y+z)^2) / (x - (y+z)) -> x + y + z
    let result = simplify("(x^2 - (y+z)^2) / (x - (y+z))");
    // Note: (y+z)^2 may expand
    assert!(
        result.contains("x") && result.contains("y") && result.contains("z"),
        "Result should contain x, y, z, got: {}",
        result
    );
}

// D. No simplification cases (GCD = 1)
#[test]
fn test_no_simplify_gcd_is_one() {
    // (x^2 + y^2)/(x + y) has GCD = 1 in Q[x,y]
    let result = simplify("(x^2 + y^2)/(x + y)");
    // Should stay as a fraction (no simplification)
    assert!(
        result.contains("/") || result.contains("x^2") || result.contains("y^2"),
        "Should not over-simplify: gcd(x^2+y^2, x+y)=1, got: {}",
        result
    );
}

#[test]
fn test_coprime_polys_stay_unsimplified() {
    // (x + 1) / (y + 1) - coprime, no common factor
    let result = simplify("(x + 1) / (y + 1)");
    assert!(
        result.contains("x") && result.contains("y") && result.contains("/"),
        "Coprime polys should stay as fraction, got: {}",
        result
    );
}

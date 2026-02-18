//! Tests for absolute value simplification rules.
//!
//! This module tests all abs-related rules in functions.rs:
//! - EvaluateAbsRule: |n| → n (numeric), |-x| → |x|
//! - AbsPositiveSimplifyRule: |x| → x when x > 0 proven
//! - AbsSquaredRule: |x|^(2k) → x^(2k)
//! - AbsIdempotentRule: ||x|| → |x|
//! - AbsOfEvenPowerRule: |x^(2k)| → x^(2k)
//! - AbsProductRule: |x|*|y| → |xy|
//! - AbsQuotientRule: |x|/|y| → |x/y|
//! - AbsSqrtRule: |sqrt(x)| → sqrt(x)
//! - AbsExpRule: |e^x| → e^x
//! - AbsSumOfSquaresRule: |x² + y²| → x² + y²

use cas_engine::Simplifier;

/// Helper to simplify an expression and return the result as a string
fn simplify(input: &str) -> String {
    let mut s = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut s.context).expect("parse failed");
    let (result, _steps) = s.simplify(expr);
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &s.context,
            id: result
        }
    )
}

/// Helper to check if simplification produces expected result
fn assert_simplifies_to(input: &str, expected: &str) {
    let result = simplify(input);
    assert_eq!(
        result, expected,
        "Simplifying '{}' expected '{}' but got '{}'",
        input, expected, result
    );
}

// =============================================================================
// EvaluateAbsRule: Numeric evaluation
// =============================================================================

#[test]
fn test_abs_positive_number() {
    assert_simplifies_to("abs(5)", "5");
}

#[test]
fn test_abs_negative_number() {
    assert_simplifies_to("abs(-5)", "5");
}

#[test]
fn test_abs_zero() {
    assert_simplifies_to("abs(0)", "0");
}

#[test]
fn test_abs_fraction() {
    assert_simplifies_to("abs(-3/4)", "3/4");
}

// =============================================================================
// EvaluateAbsRule: |-x| → |x|
// =============================================================================

#[test]
fn test_abs_of_negation() {
    // abs(-x) should simplify to abs(x)
    let result = simplify("abs(-x)");
    assert!(result == "|x|" || result == "abs(x)", "Got: {}", result);
}

// =============================================================================
// AbsIdempotentRule: ||x|| → |x|
// =============================================================================

#[test]
fn test_abs_idempotent() {
    // ||x|| should simplify to |x|
    let result = simplify("abs(abs(x))");
    assert!(result == "|x|" || result == "abs(x)", "Got: {}", result);
}

#[test]
fn test_abs_triple_nesting() {
    // |||x||| should simplify to |x|
    let result = simplify("abs(abs(abs(x)))");
    assert!(result == "|x|" || result == "abs(x)", "Got: {}", result);
}

// =============================================================================
// AbsSquaredRule: |x|^(2k) → x^(2k)
// =============================================================================

#[test]
fn test_abs_squared() {
    let result = simplify("abs(x)^2");
    // Accept x^2 or x²
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "Should remove abs: {}",
        result
    );
    assert!(
        result.contains("x") && (result.contains("2") || result.contains("²")),
        "Should be x^2: {}",
        result
    );
}

#[test]
fn test_abs_fourth_power() {
    let result = simplify("abs(x)^4");
    // Accept x^4 or x⁴
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "Should remove abs: {}",
        result
    );
    assert!(
        result.contains("x") && (result.contains("4") || result.contains("⁴")),
        "Should be x^4: {}",
        result
    );
}

#[test]
fn test_abs_odd_power_unchanged() {
    // |x|^3 should NOT simplify to x^3 (odd power)
    let result = simplify("abs(x)^3");
    assert!(
        result.contains("|") || result.contains("abs"),
        "Odd power should keep abs: {}",
        result
    );
}

// =============================================================================
// AbsOfEvenPowerRule: |x^(2k)| → x^(2k)
// =============================================================================

#[test]
fn test_abs_of_square() {
    let result = simplify("abs(x^2)");
    // Accept x^2 or x²
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "Should remove abs: {}",
        result
    );
    assert!(
        result.contains("x") && (result.contains("2") || result.contains("²")),
        "Should be x^2: {}",
        result
    );
}

#[test]
fn test_abs_of_fourth_power() {
    let result = simplify("abs(x^4)");
    // Accept x^4 or x⁴
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "Should remove abs: {}",
        result
    );
    assert!(
        result.contains("x") && (result.contains("4") || result.contains("⁴")),
        "Should be x^4: {}",
        result
    );
}

#[test]
fn test_abs_of_odd_power_unchanged() {
    // |x^3| should NOT simplify (odd power)
    let result = simplify("abs(x^3)");
    assert!(
        result.contains("|") || result.contains("abs"),
        "Odd power should keep abs: {}",
        result
    );
}

// =============================================================================
// AbsProductRule: |x|*|y| → |xy|
// =============================================================================

#[test]
fn test_abs_product() {
    let result = simplify("abs(x)*abs(y)");
    // Should be |x·y| or |x*y|
    assert!(
        result.contains("x") && result.contains("y") && result.contains("|"),
        "Product of abs should combine: {}",
        result
    );
    // Should only have one pair of | | not two
    assert_eq!(
        result.matches('|').count(),
        2,
        "Should have exactly one abs: {}",
        result
    );
}

#[test]
fn test_abs_product_three() {
    // |x|*|y|*|z| should combine progressively
    let result = simplify("abs(x)*abs(y)*abs(z)");
    // At minimum, should reduce the number of abs
    let abs_count = result.matches('|').count();
    assert!(
        abs_count <= 4,
        "Should combine abs: {} (count: {})",
        result,
        abs_count
    );
}

// =============================================================================
// AbsQuotientRule: |x|/|y| → |x/y|
// =============================================================================

#[test]
fn test_abs_quotient() {
    let result = simplify("abs(x)/abs(y)");
    // Should be |x/y| with only one pair of ||
    assert_eq!(
        result.matches('|').count(),
        2,
        "Should have exactly one abs: {}",
        result
    );
    assert!(result.contains("/"), "Should contain division: {}", result);
}

// =============================================================================
// AbsSqrtRule: |sqrt(x)| → sqrt(x)
// =============================================================================

#[test]
fn test_abs_of_sqrt_function() {
    let result = simplify("abs(sqrt(x))");
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "abs(sqrt) should simplify away: {}",
        result
    );
    // Accept √, sqrt, or x^(1/2)
    assert!(
        result.contains("√") || result.contains("sqrt") || result.contains("1/2"),
        "Should contain sqrt or power form: {}",
        result
    );
}

#[test]
fn test_abs_of_sqrt_power_form() {
    // |x^(1/2)| should simplify to x^(1/2)
    let result = simplify("abs(x^(1/2))");
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "abs(x^(1/2)) should simplify away: {}",
        result
    );
}

// =============================================================================
// AbsExpRule: |e^x| → e^x
// =============================================================================

#[test]
fn test_abs_of_exp_function() {
    let result = simplify("abs(exp(x))");
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "abs(exp) should simplify away: {}",
        result
    );
    assert!(
        result.contains("e") || result.contains("exp"),
        "Should contain exp: {}",
        result
    );
}

#[test]
fn test_abs_of_e_power() {
    // |e^x| using constant e
    let result = simplify("abs(e^x)");
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "abs(e^x) should simplify away: {}",
        result
    );
}

// =============================================================================
// AbsSumOfSquaresRule: |x² + y²| → x² + y²
// =============================================================================

#[test]
fn test_abs_of_sum_of_squares() {
    let result = simplify("abs(x^2 + y^2)");
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "abs(sum of squares) should simplify away: {}",
        result
    );
}

#[test]
fn test_abs_of_square_plus_positive() {
    // |x² + 1| should simplify (x² + 1 ≥ 1 always)
    let result = simplify("abs(x^2 + 1)");
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "abs(x² + 1) should simplify away: {}",
        result
    );
}

#[test]
fn test_abs_of_product_of_squares() {
    // |x²·y²| = x²·y²
    let result = simplify("abs(x^2 * y^2)");
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "abs(product of squares) should simplify: {}",
        result
    );
}

#[test]
fn test_abs_of_abs_squared() {
    // ||x|²| = |x|² = x²
    let result = simplify("abs(abs(x)^2)");
    // Result should be x² (no abs)
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "abs(|x|²) should simplify to x²: {}",
        result
    );
}

// =============================================================================
// Edge Cases and Combinations
// =============================================================================

#[test]
fn test_abs_of_sqrt_of_abs() {
    // |sqrt(|x|)| = sqrt(|x|)
    let result = simplify("abs(sqrt(abs(x)))");
    // Should have at most one abs (from |x|)
    let abs_count = result.matches('|').count();
    assert!(
        abs_count <= 2,
        "Should simplify outer abs: {} (count: {})",
        result,
        abs_count
    );
}

#[test]
fn test_abs_product_of_squares() {
    // |x²| * |y²| = x² * y²
    let result = simplify("abs(x^2) * abs(y^2)");
    // Should have no abs (both are even powers)
    assert!(
        !result.contains("|") && !result.contains("abs"),
        "Product of abs of squares should simplify: {}",
        result
    );
}

#[test]
fn test_complex_abs_expression() {
    // |sqrt(x)|² = x (since sqrt(x)² = x for x ≥ 0)
    let result = simplify("abs(sqrt(x))^2");
    // Should simplify to x
    assert!(result == "x", "Should simplify to x: {}", result);
}

// =============================================================================
// Non-simplifiable cases (should remain as-is)
// =============================================================================

#[test]
fn test_abs_of_variable_unchanged() {
    // |x| should remain as |x| (can't simplify without knowing sign)
    let result = simplify("abs(x)");
    assert!(
        result.contains("|") || result.contains("abs"),
        "abs(x) should remain: {}",
        result
    );
}

#[test]
fn test_abs_of_sum_unchanged() {
    // |x + y| should remain (could be negative)
    let result = simplify("abs(x + y)");
    assert!(
        result.contains("|") || result.contains("abs"),
        "abs(x+y) should remain: {}",
        result
    );
}

#[test]
fn test_abs_of_difference_unchanged() {
    // |x - y| should remain
    let result = simplify("abs(x - y)");
    assert!(
        result.contains("|") || result.contains("abs"),
        "abs(x-y) should remain: {}",
        result
    );
}

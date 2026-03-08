//! Tests for opposite denominators simplification
//!
//! These tests verify that fractions with opposite denominators
//! (e.g., x-1 and 1-x) are properly combined.

mod test_utils;

use test_utils::*;

// =============================================================================
// LEVEL 1: Simple Polynomial Cases - Symbolic Equivalence
// =============================================================================

#[test]
fn test_simple_opposite_denominators_cancels_to_zero() {
    // 1/(x-1) + 1/(1-x) = 1/(x-1) - 1/(x-1) = 0
    assert_simplifies_to_zero("1/(x-1) + 1/(1-x)");
}

#[test]
fn test_simple_opposite_denominators_with_coefficients() {
    // 2/(x-1) + 3/(1-x) = 2/(x-1) - 3/(x-1) = -1/(x-1)
    // Use numeric equivalence to avoid string format variations
    assert_equiv_numeric_1var(
        "2/(x-1) + 3/(1-x)",
        "-1/(x-1)",
        "x",
        -10.0,
        10.0,
        200,
        1e-10,
        |x| (x - 1.0).abs() > 1e-3, // Avoid singularity at x=1
    );
}

#[test]
fn test_simple_opposite_denominators_different_constant() {
    // 1/(2-x) + 1/(x-2) = 0
    assert_simplifies_to_zero("1/(2-x) + 1/(x-2)");
}

#[test]
fn test_simple_opposite_denominators_explicit_negation() {
    // -1/(1-x) + 1/(1-x) = 0
    assert_simplifies_to_zero("-1/(1-x) + 1/(1-x)");
}

#[test]
fn test_simple_opposite_denominators_larger_coefficients() {
    // 3/(x-2) + 5/(2-x) = -2/(x-2)
    assert_equiv_numeric_1var(
        "3/(x-2) + 5/(2-x)",
        "-2/(x-2)",
        "x",
        -10.0,
        10.0,
        200,
        1e-10,
        |x| (x - 2.0).abs() > 1e-3,
    );
}

// =============================================================================
// LEVEL 2: Root Cases - Numeric Equivalence (domain-sensitive)
// =============================================================================

#[test]
fn test_sqrt_opposite_denominators_cancels_numeric() {
    // 1/(sqrt(x)-1) + 1/(1-sqrt(x)) should equal 0
    // Numeric test avoids rationalization complexity
    assert_equiv_numeric_1var(
        "1/(sqrt(x)-1) + 1/(1-sqrt(x))",
        "0",
        "x",
        0.01,
        25.0,
        400,
        1e-9,
        |x| {
            x > 0.0 // sqrt needs positive
                && (x.sqrt() - 1.0).abs() > 1e-3 // avoid sqrt(x)=1
        },
    );
}

#[test]
fn test_sqrt_opposite_denominators_with_coefficients_numeric() {
    // 2/(sqrt(x)-1) + 3/(1-sqrt(x)) = -1/(sqrt(x)-1)
    assert_equiv_numeric_1var(
        "2/(sqrt(x)-1) + 3/(1-sqrt(x))",
        "-1/(sqrt(x)-1)",
        "x",
        0.01,
        25.0,
        400,
        1e-9,
        |x| x > 0.0 && (x.sqrt() - 1.0).abs() > 1e-3,
    );
}

#[test]
fn test_sqrt_with_constant_opposite_denominators_numeric() {
    // 1/(sqrt(x)+2) + 1/(-2-sqrt(x)) = 0
    assert_equiv_numeric_1var(
        "1/(sqrt(x)+2) + 1/(-2-sqrt(x))",
        "0",
        "x",
        0.01,
        25.0,
        400,
        1e-9,
        |x| {
            x > 0.0 // sqrt needs positive
                && (x.sqrt() + 2.0).abs() > 1e-3 // avoid denominator=0
        },
    );
}

// =============================================================================
// LEVEL 3: Rationalized Denominators - Numeric Equivalence
// =============================================================================

#[test]
fn test_conjugate_sum_rationalized_numeric() {
    // 1/(sqrt(x)+1) + 1/(sqrt(x)-1) = 2*sqrt(x)/(x-1)
    assert_equiv_numeric_1var(
        "1/(sqrt(x) + 1) + 1/(sqrt(x) - 1)",
        "2*sqrt(x)/(x - 1)",
        "x",
        0.01,
        25.0,
        400,
        1e-9,
        |x| {
            x > 0.0
                && (x - 1.0).abs() > 1e-3
                && (x.sqrt() - 1.0).abs() > 1e-3
                && (x.sqrt() + 1.0).abs() > 1e-3
        },
    );
}

#[test]
fn test_conjugate_difference_rationalized_numeric() {
    // 1/(sqrt(x)+1) - 1/(sqrt(x)-1) = -2/(x-1)
    assert_equiv_numeric_1var(
        "1/(sqrt(x) + 1) - 1/(sqrt(x) - 1)",
        "-2/(x - 1)",
        "x",
        0.01,
        25.0,
        400,
        1e-9,
        |x| {
            x > 0.0
                && (x - 1.0).abs() > 1e-3
                && (x.sqrt() - 1.0).abs() > 1e-3
                && (x.sqrt() + 1.0).abs() > 1e-3
        },
    );
}

// =============================================================================
// LEVEL 4: The Bridge Case - Numeric Equivalence
// =============================================================================

#[test]
fn test_bridge_case_numeric() {
    // "El Puente Conjugado": 1/(sqrt(x)+1) + 1/(sqrt(x)-1) - 2*sqrt(x)/(x-1) = 0
    assert_equiv_numeric_1var(
        "1/(sqrt(x) + 1) + 1/(sqrt(x) - 1) - (2*sqrt(x))/(x - 1)",
        "0",
        "x",
        0.01,
        25.0,
        400,
        1e-9,
        |x| {
            x > 0.0
                && (x - 1.0).abs() > 1e-3
                && (x.sqrt() - 1.0).abs() > 1e-3
                && (x.sqrt() + 1.0).abs() > 1e-3
        },
    );
}

// =============================================================================
// LEVEL 5: Mixed Cases
// =============================================================================

#[test]
fn test_three_fractions_polynomial_cancels() {
    // 1/(x-1) + 1/(1-x) + 0 = 0
    assert_simplifies_to_zero("1/(x-1) + 1/(1-x) + 0");
}

#[test]
fn test_same_denominator_combines_to_zero() {
    // 1/(x-1) + 1/(x-1) - 2/(x-1) = 0
    assert_simplifies_to_zero("1/(x-1) + 1/(x-1) - 2/(x-1)");
}

#[test]
fn test_rationalized_minus_itself_is_zero() {
    // 2*sqrt(x)/(x-1) - 2*sqrt(x)/(x-1) = 0
    assert_simplifies_to_zero("(2*sqrt(x))/(x-1) - (2*sqrt(x))/(x-1)");
}

#[test]
fn test_sqrt_numerator_opposite_denominators_numeric() {
    // sqrt(x)/(x-1) + sqrt(x)/(1-x) = 0
    assert_equiv_numeric_1var(
        "sqrt(x)/(x-1) + sqrt(x)/(1-x)",
        "0",
        "x",
        0.01,
        25.0,
        400,
        1e-9,
        |x| x > 0.0 && (x - 1.0).abs() > 1e-3,
    );
}

#[test]
fn test_expanded_bridge_parts_same_denominator() {
    // (sqrt(x)-1)/(x-1) + (sqrt(x)+1)/(x-1) - (2*sqrt(x))/(x-1) = 0
    // All denominators same, should combine symbolically
    assert_simplifies_to_zero("(sqrt(x)-1)/(x-1) + (sqrt(x)+1)/(x-1) - (2*sqrt(x))/(x-1)");
}

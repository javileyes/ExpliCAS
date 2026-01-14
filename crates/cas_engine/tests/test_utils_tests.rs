//! Tests for the test_utils module itself
//! These verify that our test helpers work correctly

mod test_utils;

use test_utils::*;

// =============================================================================
// Symbolic equivalence tests
// =============================================================================

#[test]
fn test_helper_simplifies_to_zero_pythagorean() {
    // sin²(x) + cos²(x) - 1 = 0
    assert_simplifies_to_zero("sin(x)^2 + cos(x)^2 - 1");
}

#[test]
fn test_helper_simplifies_to_one() {
    // sin²(x) + cos²(x) = 1
    assert_simplifies_to("sin(x)^2 + cos(x)^2", "1");
}

#[test]
fn test_helper_simplifies_arithmetic() {
    // 2 + 3 = 5
    assert_simplifies_to("2 + 3", "5");
}

#[test]
fn test_helper_simplifies_algebraic() {
    // x + x = 2x
    assert_simplifies_to("x + x", "2*x");
}

// =============================================================================
// Numeric equivalence tests
// =============================================================================

#[test]
fn test_helper_numeric_identity() {
    // tan(x) = sin(x)/cos(x)
    assert_equiv_numeric_1var(
        "tan(x)",
        "sin(x)/cos(x)",
        "x",
        -1.4,
        1.4, // Avoid cos(x) ≈ 0 at ±π/2
        50,
        1e-9,
        |x| x.cos().abs() > 0.1,
    );
}

#[test]
fn test_helper_numeric_double_angle() {
    // sin(2x) = 2*sin(x)*cos(x)
    assert_equiv_numeric_1var(
        "sin(2*x)",
        "2*sin(x)*cos(x)",
        "x",
        -3.0,
        3.0,
        100,
        1e-9,
        |_| true,
    );
}

#[test]
fn test_helper_numeric_pythagorean() {
    // sin²(x) + cos²(x) = 1
    assert_equiv_numeric_1var(
        "sin(x)^2 + cos(x)^2",
        "1",
        "x",
        -10.0,
        10.0,
        100,
        1e-9,
        |_| true,
    );
}

#[test]
fn test_helper_numeric_sqrt() {
    // sqrt(x^2) = |x| for x >= 0
    assert_equiv_numeric_1var(
        "sqrt(x^2)",
        "abs(x)",
        "x",
        0.1,
        10.0, // Positive values only
        50,
        1e-9,
        |x| x > 0.0,
    );
}

// =============================================================================
// 2-variable numeric tests
// =============================================================================

#[test]
fn test_helper_numeric_2var_sum() {
    // x + y = y + x (commutativity)
    assert_equiv_numeric_2var(
        "x + y",
        "y + x",
        "x",
        -5.0,
        5.0,
        "y",
        -5.0,
        5.0,
        10,
        1e-12,
        |_, _| true,
    );
}

#[test]
fn test_helper_numeric_2var_product() {
    // (x+y)^2 = x^2 + 2xy + y^2
    assert_equiv_numeric_2var(
        "(x+y)^2",
        "x^2 + 2*x*y + y^2",
        "x",
        -5.0,
        5.0,
        "y",
        -5.0,
        5.0,
        10,
        1e-9,
        |_, _| true,
    );
}

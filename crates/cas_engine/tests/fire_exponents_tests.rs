//! Fire tests: Power towers, Negative base, Exponential base, Sibling bases
//!
//! These are non-regression tests for critical exponent/power simplifications.

mod test_utils;
use test_utils::*;

// =============================================================================
// 1) Power towers: right-associativity
// =============================================================================

/// x^y^z should be interpreted as x^(y^z), not (x^y)^z
#[test]
fn power_tower_is_right_associative_in_display() {
    // The result should show x^(y^z) form
    let r = simplify_to_string("x^y^z");
    // Accept various formats: x^(y^z) or x^y^z (if already right-assoc in display)
    assert!(
        r.contains("x^(y^z)") || r == "x^(y^z)" || r == "x^y^z",
        "Power tower should display as right-associative, got: {}",
        r
    );
}

#[test]
fn power_tower_not_collapse() {
    // (x^y)^z - x^(y^z) should NOT simplify to 0
    // This catches invalid (a^b)^c → a^(bc) rules
    let r = simplify_to_string("(x^y)^z - x^(y^z)");
    assert_ne!(
        r, "0",
        "(x^y)^z should NOT equal x^(y^z) in general, got: {}",
        r
    );
}

// =============================================================================
// 2) Negative base + fractional exponent (safety)
// =============================================================================

#[test]
fn negative_base_fractional_exponent_a() {
    // ((-1)^2)^(1/2) = sqrt(1) = 1
    assert_simplifies_to("((-1)^2)^(1/2)", "1");
}

#[test]
fn negative_base_fractional_exponent_b() {
    // (-1)^(2*1/2) = (-1)^1 = -1
    assert_simplifies_to("(-1)^(2*1/2)", "-1");
}

#[test]
fn negative_base_fractional_exponent_c() {
    // ((-1)^4)^(1/4) = sqrt_4(1) = 1  (even root of 1)
    assert_simplifies_to("((-1)^4)^(1/4)", "1");
}

// =============================================================================
// 3) Exponential base reduction: (exp(x+1)/exp(1))^x → exp(x^2)
// =============================================================================

#[test]
fn exponential_base_simplification() {
    // exp(x+1)/exp(1) = exp(x), then (exp(x))^x = exp(x*x) = exp(x^2)
    // Test via numeric equivalence since form may vary
    assert_equiv_numeric_1var(
        "(exp(x + 1) / exp(1))^x",
        "exp(x^2)",
        "x",
        0.5,
        2.0,
        50,
        1e-9,
        |_| true,
    );
}

#[test]
fn exponential_quotient_simplifies() {
    // exp(x+1)/exp(1) should simplify to exp(x) via ExpQuotientRule
    assert_simplifies_to("exp(x + 1) / exp(1)", "e^(x)");
}

#[test]
fn exponential_base_reduction_symbolic() {
    // (exp(x+1)/exp(1))^x should simplify all the way to e^(x²)
    // This is the key test that motivated ExpQuotientRule
    assert_simplifies_to("(exp(x + 1) / exp(1))^x", "e^(x^2)");
}

// =============================================================================
// 4) Sibling bases: unification of powers with same base family
// =============================================================================

#[test]
fn sibling_bases_same_exponent() {
    // 2^x * 4^x * 8^x = 2^x * 2^(2x) * 2^(3x) = 2^(6x) ideally
    // CURRENT: engine produces 8^(2*x) which is equivalent but not fully collapsed
    // TODO: Add sibling base unification rule
    let r = simplify_to_string("2^x * 4^x * 8^x");
    // Accept: 2^(6x), 8^(2x), 64^x, etc. - all equivalent
    assert!(
        r.contains("^") && r.contains("x"),
        "2^x * 4^x * 8^x should produce power expression, got: {}",
        r
    );
    // Numeric equivalence ensures correctness
    assert_equiv_numeric_1var(
        "2^x * 4^x * 8^x",
        "2^(6*x)",
        "x",
        0.5,
        2.0,
        50,
        1e-9,
        |_| true,
    );
}

#[test]
fn sibling_bases_mixed_exponents_numeric() {
    // 2^x * 4^y * 8^x = 2^(4x + 2y)
    // Test via 2-var numeric equivalence
    assert_equiv_numeric_2var(
        "2^x * 4^y * 8^x",
        "2^(4*x + 2*y)",
        "x",
        0.5,
        2.0,
        "y",
        0.5,
        2.0,
        20,
        1e-9,
        |_, _| true,
    );
}

#[test]
fn sibling_bases_3_and_9() {
    // 3^x * 9^x = 3^x * 3^(2x) = 3^(3x)
    let r = simplify_to_string("3^x * 9^x");
    assert!(
        r.contains("3^(3") || r.contains("27^x") || r.contains("3^3"),
        "3^x * 9^x should collapse to 3^(3x) or equivalent, got: {}",
        r
    );
}

// =============================================================================
// 5) Power of power with even root (parity handling from V2.14.45)
// =============================================================================

#[test]
fn power_of_power_even_root() {
    // (x^2)^(1/2) = |x|
    assert_simplifies_to("(x^2)^(1/2)", "|x|");
}

#[test]
fn power_of_power_4th_root() {
    // (x^4)^(1/4) = |x|
    assert_simplifies_to("(x^4)^(1/4)", "|x|");
}

#[test]
fn power_of_power_odd_root() {
    // (x^3)^(1/3) = x
    assert_simplifies_to("(x^3)^(1/3)", "x");
}

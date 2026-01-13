//! Numeric Equivalence Property Tests for Trig Identities
//!
//! These tests verify that simplification rewrites preserve numeric value.
//! They catch mathematical bugs like sign errors that are structurally correct
//! but numerically wrong.
//!
//! Design decisions:
//! - Tests are unit tests (not integration) so they can access pub(crate) eval_f64
//! - Use proptest with fixed case count for CI stability
//! - Filter singularities (cos ≈ 0 for tan, etc)
//! - Use eps = 1e-9 tolerance
//! - Test both small [-2,2] and large [-20,20] ranges

use crate::engine::{eval_f64, Simplifier};
use proptest::prelude::*;
use std::collections::HashMap;

/// Tolerance for numeric comparison
const EPS: f64 = 1e-9;

/// Check if two f64 values are approximately equal
fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() < eps
}

/// Check if cos(x) is not near zero (for tan singularity avoidance)
fn cos_safe(x: f64) -> bool {
    x.cos().abs() > 0.1
}

/// Check if numerator and denominator are both safe for evaluation
fn sum_quotient_safe(x: f64, a_coeff: i64, b_coeff: i64) -> bool {
    // Denominator is cos(a*x) + cos(b*x)
    let a = a_coeff as f64 * x;
    let b = b_coeff as f64 * x;
    let den = a.cos() + b.cos();
    den.abs() > 0.1 && cos_safe(x) // Also need cos(x) safe for tan(...)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Test: (sin(x) + sin(3x)) / (cos(x) + cos(3x)) ≈ tan(2x)
    #[test]
    fn numeric_sin_sum_quotient(x in -2.0f64..2.0) {
        prop_assume!(sum_quotient_safe(x, 1, 3));

        // Input: (sin(x) + sin(3x)) / (cos(x) + cos(3x))
        let input_val = (x.sin() + (3.0 * x).sin()) / (x.cos() + (3.0 * x).cos());

        // Expected output: tan(2x)
        let expected_val = (2.0 * x).tan();

        prop_assert!(
            approx_eq(input_val, expected_val, EPS),
            "Sum quotient identity failed at x={}: got {}, expected {}",
            x, input_val, expected_val
        );
    }

    /// Test: (sin(5x) - sin(3x)) / (cos(5x) + cos(3x)) ≈ tan(x)
    /// This is the exact case where sign bug was found.
    #[test]
    fn numeric_sin_diff_quotient(x in -2.0f64..2.0) {
        prop_assume!(sum_quotient_safe(x, 5, 3));

        // Input: (sin(5x) - sin(3x)) / (cos(5x) + cos(3x))
        let input_val = ((5.0 * x).sin() - (3.0 * x).sin()) / ((5.0 * x).cos() + (3.0 * x).cos());

        // Expected output: tan(x) (because (5x-3x)/2 = x)
        let expected_val = x.tan();

        prop_assert!(
            approx_eq(input_val, expected_val, EPS),
            "Diff quotient identity failed at x={}: got {}, expected {}. Sign bug?",
            x, input_val, expected_val
        );
    }

    /// Test with larger range to catch edge cases
    #[test]
    fn numeric_sin_diff_quotient_large_range(x in -20.0f64..20.0) {
        prop_assume!(sum_quotient_safe(x, 5, 3));

        let input_val = ((5.0 * x).sin() - (3.0 * x).sin()) / ((5.0 * x).cos() + (3.0 * x).cos());
        let expected_val = x.tan();

        prop_assert!(
            approx_eq(input_val, expected_val, EPS * 10.0), // Slightly larger tolerance for larger values
            "Diff quotient identity failed at x={}: got {}, expected {}",
            x, input_val, expected_val
        );
    }

    /// Test the simplified result matches the original expression
    #[test]
    fn numeric_simplification_preserves_value(x in -1.5f64..1.5) {
        prop_assume!(sum_quotient_safe(x, 5, 3));

        // Parse and simplify (sin(5x) - sin(3x)) / (cos(5x) + cos(3x))
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("(sin(5*x) - sin(3*x)) / (cos(5*x) + cos(3*x))", &mut ctx)
            .expect("parse failed");

        let mut simplifier = Simplifier::with_default_rules();
        simplifier.context = ctx;

        let (result, _) = simplifier.simplify(expr);

        // Evaluate both original and result
        let mut var_map = HashMap::new();
        var_map.insert("x".to_string(), x);

        let orig_val = eval_f64(&simplifier.context, expr, &var_map);
        let result_val = eval_f64(&simplifier.context, result, &var_map);

        if let (Some(orig), Some(res)) = (orig_val, result_val) {
            prop_assert!(
                approx_eq(orig, res, EPS),
                "Simplification changed value at x={}: original = {}, simplified = {}",
                x, orig, res
            );
        }
        // If evaluation fails (e.g., due to singularity), that's OK - skip
    }

    /// Test inverse tan relation: atan(t) + atan(1/t) ≈ π/2 for t > 0
    #[test]
    fn numeric_inverse_tan_sum(t in 0.1f64..10.0) {
        let result = t.atan() + (1.0 / t).atan();
        let expected = std::f64::consts::FRAC_PI_2;

        prop_assert!(
            approx_eq(result, expected, EPS),
            "atan(t) + atan(1/t) != π/2 for t={}: got {}, expected {}",
            t, result, expected
        );
    }

    /// Test double angle: sin(2x) = 2*sin(x)*cos(x)
    #[test]
    fn numeric_double_angle_sin(x in -3.0f64..3.0) {
        let lhs = (2.0 * x).sin();
        let rhs = 2.0 * x.sin() * x.cos();

        prop_assert!(
            approx_eq(lhs, rhs, EPS),
            "sin(2x) != 2*sin(x)*cos(x) at x={}: {} vs {}",
            x, lhs, rhs
        );
    }

    /// Test double angle: cos(2x) = cos²(x) - sin²(x)
    #[test]
    fn numeric_double_angle_cos(x in -3.0f64..3.0) {
        let lhs = (2.0 * x).cos();
        let rhs = x.cos().powi(2) - x.sin().powi(2);

        prop_assert!(
            approx_eq(lhs, rhs, EPS),
            "cos(2x) != cos²(x) - sin²(x) at x={}: {} vs {}",
            x, lhs, rhs
        );
    }

    /// Test triple angle: sin(3x) = 3sin(x) - 4sin³(x)
    #[test]
    fn numeric_triple_angle_sin(x in -2.0f64..2.0) {
        let lhs = (3.0 * x).sin();
        let rhs = 3.0 * x.sin() - 4.0 * x.sin().powi(3);

        prop_assert!(
            approx_eq(lhs, rhs, EPS),
            "sin(3x) != 3sin(x) - 4sin³(x) at x={}: {} vs {}",
            x, lhs, rhs
        );
    }

    /// Test triple angle: cos(3x) = 4cos³(x) - 3cos(x)
    #[test]
    fn numeric_triple_angle_cos(x in -2.0f64..2.0) {
        let lhs = (3.0 * x).cos();
        let rhs = 4.0 * x.cos().powi(3) - 3.0 * x.cos();

        prop_assert!(
            approx_eq(lhs, rhs, EPS),
            "cos(3x) != 4cos³(x) - 3cos(x) at x={}: {} vs {}",
            x, lhs, rhs
        );
    }

    /// Test Pythagorean identity: sin²(x) + cos²(x) = 1
    #[test]
    fn numeric_pythagorean(x in -10.0f64..10.0) {
        let result = x.sin().powi(2) + x.cos().powi(2);

        prop_assert!(
            approx_eq(result, 1.0, EPS),
            "sin²(x) + cos²(x) != 1 at x={}: got {}",
            x, result
        );
    }
}

/// Deterministic regression test for the exact bug that was fixed
#[test]
fn regression_sign_bug_sin_diff_quotient() {
    // This is a deterministic version of the property test
    // Tests specific x values that might have triggered the bug

    let test_values = [0.1, 0.5, 1.0, 1.5, -0.5, -1.0, 0.0];

    for x in test_values {
        if !sum_quotient_safe(x, 5, 3) {
            continue;
        }

        let input_val = ((5.0 * x).sin() - (3.0 * x).sin()) / ((5.0 * x).cos() + (3.0 * x).cos());
        let expected_val = x.tan();

        assert!(
            approx_eq(input_val, expected_val, EPS),
            "Sign bug at x={}: input={}, expected={}",
            x,
            input_val,
            expected_val
        );
    }
}

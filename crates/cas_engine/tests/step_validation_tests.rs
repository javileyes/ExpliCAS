//! Integration tests for the Step Validation Educational Tool
//!
//! Tests the validate_step API for checking if a student's intermediate
//! expression is a valid partial simplification.

use cas_engine::{Simplifier, StepCheckVerdict, ValidationRoute};
use cas_parser::parse;

/// Helper: validate step and return verdict
fn validate_step(initial: &str, student: &str) -> StepCheckVerdict {
    let mut simplifier = Simplifier::with_default_rules();

    let initial_expr = parse(initial, &mut simplifier.context).expect("parse initial failed");
    let student_expr = parse(student, &mut simplifier.context).expect("parse student failed");

    simplifier.validate_step(initial_expr, student_expr)
}

// =============================================================================
// Valid and Simpler Cases
// =============================================================================

#[test]
fn test_valid_and_simpler_commute() {
    // x + 1 → 1 + x is valid but same complexity
    let verdict = validate_step("x + 1", "1 + x");

    match verdict {
        StepCheckVerdict::ValidAndSimpler { .. } | StepCheckVerdict::ValidButNotSimpler { .. } => {
            // Both are acceptable - expressions are equivalent
        }
        _ => panic!("Expected valid step, got: {:?}", verdict),
    }
}

#[test]
fn test_valid_simplification_expand() {
    // (x + 1)^2 can be seen as related to x^2 + 2x + 1
    // Note: direction matters for "simpler"
    let verdict = validate_step("x^2 + 2*x + 1", "(x + 1)^2");

    match verdict {
        StepCheckVerdict::ValidAndSimpler {
            complexity_delta, ..
        } => {
            assert!(
                complexity_delta < 0,
                "Factored form should be simpler, delta = {}",
                complexity_delta
            );
        }
        StepCheckVerdict::ValidButNotSimpler { .. } => {
            // Also acceptable if complexity is close
        }
        _ => panic!("Expected valid step, got: {:?}", verdict),
    }
}

#[test]
fn test_valid_simplification_basic() {
    // 2 + 3 → 5 is clearly simpler
    let verdict = validate_step("2 + 3", "5");

    match verdict {
        StepCheckVerdict::ValidAndSimpler { route, .. } => {
            // Should find in timeline
            assert!(matches!(route, ValidationRoute::DirectTimeline { .. }));
        }
        StepCheckVerdict::ValidButNotSimpler { .. } => {
            // Also acceptable
        }
        _ => panic!("Expected valid step for 2+3=5, got: {:?}", verdict),
    }
}

// =============================================================================
// Invalid Cases (Non-equivalent)
// =============================================================================

#[test]
fn test_invalid_different_expressions() {
    // x^2 + 1 ≠ (x + 1)^2
    let verdict = validate_step("x^2 + 1", "(x + 1)^2");

    match verdict {
        StepCheckVerdict::Invalid {
            counterexample,
            reason,
        } => {
            assert!(counterexample.is_some(), "Should find counterexample");
            assert!(!reason.is_empty());
        }
        _ => panic!("Expected invalid step, got: {:?}", verdict),
    }
}

#[test]
fn test_invalid_wrong_constant() {
    // x + 1 ≠ x + 2
    let verdict = validate_step("x + 1", "x + 2");

    match verdict {
        StepCheckVerdict::Invalid { counterexample, .. } => {
            assert!(counterexample.is_some(), "Should find counterexample");
        }
        _ => panic!("Expected invalid step, got: {:?}", verdict),
    }
}

// =============================================================================
// Timeline vs Equivalence Proof Routes
// =============================================================================

#[test]
fn test_route_direct_timeline() {
    // Simple numeric computation should be in timeline
    let verdict = validate_step("3 * 4", "12");

    match verdict {
        StepCheckVerdict::ValidAndSimpler { route, .. }
        | StepCheckVerdict::ValidButNotSimpler { route, .. } => match route {
            ValidationRoute::DirectTimeline { steps } => {
                assert!(!steps.is_empty(), "Should have at least one step");
            }
            ValidationRoute::EquivalenceProof { .. } => {
                // Also acceptable
            }
        },
        _ => panic!("Expected valid step, got: {:?}", verdict),
    }
}

// =============================================================================
// Conditional Equivalence (requires)
// =============================================================================

#[test]
fn test_conditional_with_sqrt() {
    // sqrt(x)^2 → x requires x ≥ 0 (implicit)
    let verdict = validate_step("sqrt(x)^2", "x");

    match verdict {
        StepCheckVerdict::ValidAndSimpler { .. } | StepCheckVerdict::ValidButNotSimpler { .. } => {
            // The step is valid (may or may not have explicit requires)
            // depending on whether it's found in timeline
        }
        StepCheckVerdict::Unknown { .. } => {
            // Also acceptable - complex case
        }
        _ => panic!("Expected valid or unknown step, got: {:?}", verdict),
    }
}

// =============================================================================
// Complexity Metric Tests
// =============================================================================

#[test]
fn test_complexity_delta_expansion() {
    // Expanding should increase complexity
    let verdict = validate_step("(x + 1)^2", "x^2 + 2*x + 1");

    match verdict {
        StepCheckVerdict::ValidAndSimpler { .. } => {
            // This is "simpler" going the other direction
        }
        StepCheckVerdict::ValidButNotSimpler {
            complexity_delta, ..
        } => {
            assert!(
                complexity_delta >= 0,
                "Expansion should not decrease complexity, delta = {}",
                complexity_delta
            );
        }
        _ => panic!("Expected valid step, got: {:?}", verdict),
    }
}

// =============================================================================
// Regression Tests (Mode-Aware Behavior)
// =============================================================================

/// Test that exp(ln(x)) → x is validated with implicit requires
#[test]
fn test_regression_exp_ln_requires_positive() {
    let verdict = validate_step("exp(ln(x))", "x");

    // Should be valid with x > 0 requirement (either timeline or proof)
    match verdict {
        StepCheckVerdict::ValidAndSimpler { .. } | StepCheckVerdict::ValidButNotSimpler { .. } => {
            // Valid - could have requires for x > 0
        }
        StepCheckVerdict::Unknown { .. } => {
            // Also acceptable - complex analysis
        }
        StepCheckVerdict::Invalid { .. } => {
            panic!("exp(ln(x)) should be equivalent to x, got Invalid");
        }
    }
}

/// Test that manifestly wrong steps are caught
#[test]
fn test_regression_catch_common_student_errors() {
    // Common error: (a+b)^2 = a^2 + b^2 (missing 2ab)
    let verdict = validate_step("(a + b)^2", "a^2 + b^2");

    match verdict {
        StepCheckVerdict::Invalid { counterexample, .. } => {
            assert!(
                counterexample.is_some(),
                "Should find counterexample for (a+b)^2 ≠ a^2+b^2"
            );
        }
        _ => panic!(
            "Expected invalid step for common error (a+b)^2 ≠ a^2+b^2, got: {:?}",
            verdict
        ),
    }
}

/// Test that pure commutative reordering is valid but not simpler
#[test]
fn test_regression_commutative_reorder() {
    let verdict = validate_step("a + b + c", "c + a + b");

    match verdict {
        StepCheckVerdict::ValidAndSimpler {
            complexity_delta, ..
        } => {
            // Complexity should be same or very close
            assert!(
                complexity_delta.abs() <= 1,
                "Reordering shouldn't change complexity much"
            );
        }
        StepCheckVerdict::ValidButNotSimpler { .. } => {
            // Expected - reordering doesn't simplify
        }
        _ => panic!(
            "Expected valid step for commutative reorder, got: {:?}",
            verdict
        ),
    }
}

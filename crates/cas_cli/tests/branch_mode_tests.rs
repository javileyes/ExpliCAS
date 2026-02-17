//! Tests for BranchMode feature (strict vs principal branch)
//!
//! NOTE: These tests now use SimplifyOptions with inv_trig policy.
//! PrincipalBranchInverseTrigRule is self-gated by inv_trig_policy().

use cas_formatter::DisplayExpr;
use cas_solver::{InverseTrigPolicy, Simplifier, SimplifyOptions};

fn simplify_strict(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");
    // Default SimplifyOptions has inv_trig = Strict
    let opts = SimplifyOptions::default();
    let (result, _, _) = simplifier.simplify_with_stats(expr, opts);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
    )
}

fn simplify_principal_branch(input: &str) -> (String, Vec<String>) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");
    // Set inv_trig = PrincipalValue to enable principal branch rules
    let opts = SimplifyOptions {
        shared: cas_solver::phase::SharedSemanticConfig {
            semantics: cas_solver::semantics::EvalConfig {
                inv_trig: InverseTrigPolicy::PrincipalValue,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let (result, steps, _) = simplifier.simplify_with_stats(expr, opts);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
    );
    // Collect domain assumptions from all steps (via assumption_events)
    let assumptions: Vec<_> = steps
        .iter()
        .flat_map(|s| s.assumption_events())
        .map(|e| e.message.clone())
        .collect();
    (result_str, assumptions)
}

// ============================================================================
// Strict Mode Tests (Default - Safe)
// ============================================================================

#[test]
fn test_strict_arctan_tan_x_unchanged() {
    // In strict mode, arctan(tan(x)) should NOT simplify to x
    let result = simplify_strict("arctan(tan(x))");
    assert!(
        result.contains("arctan") || result.contains("atan") || result.contains("sin"),
        "Strict mode should NOT simplify arctan(tan(x)), got: {}",
        result
    );
}

#[test]
fn test_strict_arcsin_sin_x_unchanged() {
    // In strict mode, arcsin(sin(x)) should NOT simplify to x
    let result = simplify_strict("arcsin(sin(x))");
    assert!(
        result.contains("arcsin") || result.contains("asin") || result.contains("sin"),
        "Strict mode should NOT simplify arcsin(sin(x)), got: {}",
        result
    );
}

#[test]
fn test_strict_arccos_cos_x_unchanged() {
    // In strict mode, arccos(cos(x)) should NOT simplify to x
    let result = simplify_strict("arccos(cos(x))");
    assert!(
        result.contains("arccos") || result.contains("acos") || result.contains("cos"),
        "Strict mode should NOT simplify arccos(cos(x)), got: {}",
        result
    );
}

// ============================================================================
// Principal Branch Mode Tests (Educational)
// ============================================================================

#[test]
fn test_principal_arctan_tan_x_simplifies() {
    // In principal branch mode, arctan(tan(x)) → x
    let (result, assumptions) = simplify_principal_branch("arctan(tan(x))");
    assert_eq!(
        result, "x",
        "Principal branch mode should simplify arctan(tan(x)) to x"
    );
    assert!(
        !assumptions.is_empty(),
        "Should emit domain assumption warning"
    );
}

#[test]
fn test_principal_arcsin_sin_x_simplifies() {
    // In principal branch mode, arcsin(sin(x)) → x
    let (result, assumptions) = simplify_principal_branch("arcsin(sin(x))");
    assert_eq!(
        result, "x",
        "Principal branch mode should simplify arcsin(sin(x)) to x"
    );
    assert!(
        !assumptions.is_empty(),
        "Should emit domain assumption warning"
    );
}

#[test]
fn test_principal_arccos_cos_x_simplifies() {
    // In principal branch mode, arccos(cos(x)) → x
    let (result, assumptions) = simplify_principal_branch("arccos(cos(x))");
    assert_eq!(
        result, "x",
        "Principal branch mode should simplify arccos(cos(x)) to x"
    );
    assert!(
        !assumptions.is_empty(),
        "Should emit domain assumption warning"
    );
}

// ============================================================================
// Educational Counterexample Tests
// ============================================================================

#[test]
fn test_principal_arctan_tan_pi_evaluates_correctly() {
    // arctan(tan(pi)) → 0 because tan(pi) = 0 evaluates FIRST, then arctan(0) = 0
    // This is actually mathematically correct! The principal branch rule doesn't
    // fire here because the inner tan(pi) is evaluated before the composition is seen.
    let (result, _assumptions) = simplify_principal_branch("arctan(tan(pi))");
    assert_eq!(
        result, "0",
        "arctan(tan(pi)) = arctan(0) = 0 (tan(pi) evaluates first)"
    );
}

#[test]
fn test_principal_arctan_tan_y_with_warning() {
    // arctan(tan(y)) → y in principal mode with warning
    // This demonstrates the educational point: y may exceed (-π/2, π/2)
    let (result, assumptions) = simplify_principal_branch("arctan(tan(y))");
    assert_eq!(
        result, "y",
        "Principal branch mode simplifies arctan(tan(y)) to y"
    );
    assert!(
        !assumptions.is_empty(),
        "Must emit domain warning since y may exceed principal domain"
    );
}

// ============================================================================
// Safe Compositions (should work in BOTH modes)
// ============================================================================

#[test]
fn test_both_modes_tan_arctan_x() {
    // tan(arctan(x)) → x should work in BOTH modes (function∘inverse is always safe)
    let strict = simplify_strict("tan(arctan(x))");
    let (principal, _) = simplify_principal_branch("tan(arctan(x))");

    assert_eq!(strict, "x", "tan(arctan(x)) → x in strict mode");
    assert_eq!(principal, "x", "tan(arctan(x)) → x in principal mode");
}

#[test]
fn test_both_modes_sin_arcsin_x() {
    // sin(arcsin(x)) → x should work in BOTH modes
    let strict = simplify_strict("sin(arcsin(x))");
    let (principal, _) = simplify_principal_branch("sin(arcsin(x))");

    assert_eq!(strict, "x", "sin(arcsin(x)) → x in strict mode");
    assert_eq!(principal, "x", "sin(arcsin(x)) → x in principal mode");
}

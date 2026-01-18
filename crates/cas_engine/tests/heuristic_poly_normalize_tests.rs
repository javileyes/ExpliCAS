//! Unit tests for HeuristicPolyNormalizeAddRule behavior
//!
//! V2.15.8: These tests ensure Heuristic mode:
//! - Normalizes Add/Sub with Pow(Add, n) for n ≤ 6
//! - Does NOT touch expressions that exceed budget or are non-polynomial
//!
//! This prevents "silent widening" of Heuristic mode in future changes.

use cas_ast::Context;
use cas_engine::options::AutoExpandBinomials;
use cas_engine::phase::SimplifyOptions;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper to simplify with specific autoexpand_binomials mode
fn simplify_with_mode(input: &str, mode: AutoExpandBinomials) -> String {
    let mut ctx = Context::new();
    let parsed = parse(input, &mut ctx).expect("parse failed");
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;

    let opts = SimplifyOptions {
        autoexpand_binomials: mode,
        ..Default::default()
    };

    let (result, _steps) = simplifier.simplify_with_options(parsed, opts);
    format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// =============================================================================
// Tests: Should normalize (combine like terms in additive context)
// =============================================================================

#[test]
fn heuristic_normalizes_binomial_cube_plus_cube() {
    // (x+1)^3 + x^3 → 2x³ + 3x² + 3x + 1
    let result = simplify_with_mode("(x+1)^3 + x^3", AutoExpandBinomials::Heuristic);

    // Verify key terms are present (coefficients 2, 3 should appear)
    assert!(
        result.contains("2"),
        "Expected coefficient 2 for x³: {}",
        result
    );
    assert!(
        !result.contains("(x + 1)"),
        "Should not contain unexpanded (x+1): {}",
        result
    );
}

#[test]
fn heuristic_normalizes_pow4_plus_multiple() {
    // (x+1)^4 + 4*(x+1)^3 → x⁴ + 8x³ + 18x² + 16x + 5
    let result = simplify_with_mode("(x+1)^4 + 4*(x+1)^3", AutoExpandBinomials::Heuristic);

    // Verify the constant term is 5 (not 4)
    assert!(
        result.ends_with("5") || result.contains("+ 5"),
        "Expected constant term 5: {}",
        result
    );
    assert!(
        !result.contains("(x + 1)"),
        "Should not contain unexpanded (x+1): {}",
        result
    );
}

#[test]
fn heuristic_identity_equals_zero() {
    // (x+1)^5 - expansion = 0
    let result = simplify_with_mode(
        "(x+1)^5 - (x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1)",
        AutoExpandBinomials::Heuristic,
    );
    assert_eq!(result, "0", "Identity should simplify to 0: {}", result);
}

// =============================================================================
// Tests: Should NOT touch (budget exceeded or non-polynomial)
// =============================================================================

#[test]
fn heuristic_skips_pow7_budget_exceeded() {
    // (x+1)^7 + x^7 → unchanged (n > 6, exceeds budget)
    let result = simplify_with_mode("(x+1)^7 + x^7", AutoExpandBinomials::Heuristic);

    // Should still contain (x+1)^7 in some form
    assert!(
        result.contains("(x + 1)") || result.contains("x + 1"),
        "Should preserve (x+1)^7 structure (budget exceeded): {}",
        result
    );
}

#[test]
fn heuristic_skips_non_polynomial_sin() {
    // (x+sin(x))^3 + x^3 → unchanged (not polynomial, contains sin)
    let result = simplify_with_mode("(x + sin(x))^3 + x^3", AutoExpandBinomials::Heuristic);

    // Should still contain sin function
    assert!(
        result.contains("sin"),
        "Should preserve sin function (not polynomial): {}",
        result
    );
}

#[test]
fn heuristic_skips_non_polynomial_sqrt() {
    // (x+sqrt(x))^3 + x^3 → unchanged (not polynomial, contains sqrt)
    // Using ^3 because ^2 is expanded by AutoExpandPowSumRule (cheap binomial expansion)
    // This is a guard test to prevent sqrt being treated as polynomial by HeuristicPolyNormalizeAddRule
    let result = simplify_with_mode("(x + sqrt(x))^3 + x^3", AutoExpandBinomials::Heuristic);

    // Should still contain sqrt function (not fully polynomial-expanded)
    // sqrt(x) can be represented as √(x), sqrt(x), or x^(1/2)
    assert!(
        result.contains("√") || result.contains("sqrt") || result.contains("^(1/2)"),
        "Should preserve sqrt function (not polynomial): {}",
        result
    );
}

// =============================================================================
// Tests: Mode comparison (Off vs Heuristic vs On)
// =============================================================================

#[test]
fn off_mode_preserves_structure() {
    // Off mode should NOT expand in additive context
    let result = simplify_with_mode("(x+1)^3 + x^3", AutoExpandBinomials::Off);

    // Should still contain (x+1)^3
    assert!(
        result.contains("(x + 1)"),
        "Off mode should preserve (x+1)³ structure: {}",
        result
    );
}

#[test]
fn heuristic_standalone_binomial_unchanged() {
    // Standalone (x+1)^5 should NOT be expanded in Heuristic mode
    let result = simplify_with_mode("(x+1)^5", AutoExpandBinomials::Heuristic);

    // Should still be in binomial form
    assert!(
        result.contains("(x + 1)") || result.contains("x + 1"),
        "Heuristic should preserve standalone (x+1)⁵: {}",
        result
    );
}

#[test]
fn on_mode_expands_standalone() {
    // On mode should expand even standalone binomials
    let result = simplify_with_mode("(x+1)^3", AutoExpandBinomials::On);

    // Should be fully expanded (no longer contain just (x+1)^3)
    assert!(
        !result.contains("(x + 1)³") && !result.contains("(x + 1)^3"),
        "On mode should expand (x+1)³: {}",
        result
    );
}

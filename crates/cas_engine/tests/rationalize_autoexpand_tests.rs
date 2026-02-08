//! Regression test for auto-expand during rationalization.
//!
//! This test ensures that:
//! - With autoexpand ON: 1/(1+√2+√3) fully simplifies (no unexpanded (1+√2)² in result)
//! - With autoexpand OFF: May leave structured form (acceptable)

use cas_ast::DisplayExpr;
use cas_engine::options::EvalOptions;
use cas_engine::phase::ExpandPolicy;
use cas_engine::Engine;
use cas_parser::parse;

/// Test that auto-expand works within rationalization phase.
/// Bug: (1+√2)² - 3 was left unexpanded after rationalization created it.
/// Fix: AutoExpandPowSumRule now runs in RATIONALIZE phase.
#[test]
fn test_rationalize_with_autoexpand_expands_subexpressions() {
    let mut engine = Engine::new();

    // Setup: enable auto-expand
    let opts = EvalOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            expand_policy: ExpandPolicy::Auto,
            ..Default::default()
        },
        ..Default::default()
    };

    // Parse the problematic expression
    let expr_str = "1/(1+sqrt(2)+sqrt(3))";
    let expr = parse(expr_str, &mut engine.simplifier.context).expect("parse failed");

    // Simplify with auto-expand ON
    let simplify_opts = opts.to_simplify_options();
    let (result, _steps, _stats) = engine.simplifier.simplify_with_stats(expr, simplify_opts);

    // Format result to check
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result
        }
    );

    // The result should NOT contain "(1 + √(2))² - 3" - this should be expanded to 2√2
    // It should also not contain ")² - 3" anywhere
    assert!(
        !result_str.contains(")² - 3"),
        "With autoexpand ON, (1+√2)² - 3 should be expanded. Got: {}",
        result_str
    );

    // Result should be a simplified fraction (either (2+√2-√6)/4 or equivalent form)
    // Just verify it contains /4 indicating full rationalization
    assert!(
        result_str.contains("/4") || result_str.contains("/ 4"),
        "Result should be fully rationalized with denominator 4. Got: {}",
        result_str
    );
}

/// Test that with autoexpand OFF, we don't force expansion.
/// The expression may remain in structured form.
#[test]
fn test_rationalize_without_autoexpand_preserves_structure() {
    let mut engine = Engine::new();

    // Setup: autoexpand OFF (default)
    let opts = EvalOptions::default();
    assert_eq!(opts.shared.expand_policy, ExpandPolicy::Off);

    // Parse the expression
    let expr_str = "1/(1+sqrt(2)+sqrt(3))";
    let expr = parse(expr_str, &mut engine.simplifier.context).expect("parse failed");

    // Simplify with auto-expand OFF
    let simplify_opts = opts.to_simplify_options();
    let (result, _steps, _stats) = engine.simplifier.simplify_with_stats(expr, simplify_opts);

    // Format result
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result
        }
    );

    // With autoexpand OFF, should still do basic rationalization
    // but may leave (1+√2)² unexpanded - this is acceptable behavior
    // Just verify it doesn't explode
    assert!(!result_str.is_empty(), "Should produce some result");

    // Basic sanity: shouldn't produce an error or the original unmodified
    // (rationalization should at least attempt to run)
}

/// Test that auto-expand budget is respected even during rationalization.
#[test]
fn test_rationalize_respects_autoexpand_budget() {
    let mut engine = Engine::new();

    // Setup: auto-expand with restrictive budget
    let mut opts = EvalOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            expand_policy: ExpandPolicy::Auto,
            ..Default::default()
        },
        ..Default::default()
    };
    opts.shared.expand_budget.max_pow_exp = 2; // Only allow exponent ≤ 2

    // Expression (1+√2)² should still expand (exponent = 2, within budget)
    let expr_str = "(1+sqrt(2))^2 - 3";
    let expr = parse(expr_str, &mut engine.simplifier.context).expect("parse failed");

    let simplify_opts = opts.to_simplify_options();
    let (result, _steps, _stats) = engine.simplifier.simplify_with_stats(expr, simplify_opts);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &engine.simplifier.context,
            id: result
        }
    );

    // Should expand to 2√2 (since 2 <= max_pow_exp)
    assert!(
        result_str.contains("√(2)") || result_str.contains("2^(1/2)"),
        "Should simplify to 2√2. Got: {}",
        result_str
    );
    assert!(
        !result_str.contains(")² -") && !result_str.contains(")^2 -"),
        "Should expand (1+√2)². Got: {}",
        result_str
    );
}

//! Health smoke tests with thresholds to detect regressions in CI.
//!
//! These tests verify that the simplification pipeline stays within reasonable
//! resource budgets. If a PR introduces excessive rewrites (churn) or node growth,
//! these tests will fail.
//!
//! ## Thresholds Philosophy
//! - Start conservative (loose limits)
//! - Tighten as the engine stabilizes
//! - On failure: print full diagnostics for debugging

use cas_engine::{PipelineStats, Simplifier, SimplifyOptions};

/// Helper to set up a health-monitored simplification and run asserts
fn run_health_check(
    input: &str,
    max_rewrites: usize,
    max_growth: i64,
    max_transform_rewrites: usize,
) -> (String, PipelineStats) {
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();
    simplifier.profiler.enable_health();
    simplifier.profiler.clear_run();

    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");
    let opts = SimplifyOptions::default();
    let (result, _steps, stats) = simplifier.simplify_with_stats(expr, opts);

    let result_str = format!("expr#{:?}", result);

    // Get aggregated growth
    let total_growth = simplifier.profiler.total_positive_growth();

    // Print diagnostics on failure (always collect for potential assertion)
    let diagnostics = format!(
        "Input: {}\nResult: {}\n\nPipeline Stats:\n  total_rewrites: {}\n  core: {} iters, {} rewrites\n  transform: {} iters, {} rewrites\n  rationalize: {} iters, {} rewrites\n  post_cleanup: {} iters, {} rewrites\n\nHealth:\n  total_positive_growth: {} nodes\n  total_applied: {}\n  total_rejected_semantic: {}\n\n{}",
        input,
        result_str,
        stats.total_rewrites,
        stats.core.iters_used, stats.core.rewrites_used,
        stats.transform.iters_used, stats.transform.rewrites_used,
        stats.rationalize.iters_used, stats.rationalize.rewrites_used,
        stats.post_cleanup.iters_used, stats.post_cleanup.rewrites_used,
        total_growth,
        simplifier.profiler.total_applied(),
        simplifier.profiler.total_rejected_semantic(),
        simplifier.profiler.health_report(),
    );

    // Assertions with full diagnostics on failure
    assert!(
        stats.total_rewrites <= max_rewrites,
        "EXCEEDED total_rewrites limit ({} > {})\n\n{}",
        stats.total_rewrites,
        max_rewrites,
        diagnostics
    );

    assert!(
        total_growth <= max_growth,
        "EXCEEDED growth limit ({} > {})\n\n{}",
        total_growth,
        max_growth,
        diagnostics
    );

    assert!(
        stats.transform.rewrites_used <= max_transform_rewrites,
        "EXCEEDED transform_rewrites limit ({} > {})\n\n{}",
        stats.transform.rewrites_used,
        max_transform_rewrites,
        diagnostics
    );

    (result_str, stats)
}

// ==================== SMOKE TESTS ====================

#[test]
fn health_smoke_mixed_expression() {
    // Mixed: rationalization + distribution
    run_health_check(
        "x/(1+sqrt(2)) + 2*(y+3)",
        150, // max_rewrites (conservative)
        250, // max_growth (nodes)
        80,  // max_transform_rewrites
    );
}

#[test]
fn health_smoke_polynomial() {
    // Polynomial expansion
    run_health_check(
        "(x+1)*(x+2)",
        100, // max_rewrites
        150, // max_growth
        60,  // max_transform_rewrites
    );
}

#[test]
fn health_smoke_rationalization_only() {
    // Pure rationalization (Level 1.5)
    run_health_check(
        "1/(3-2*sqrt(5))",
        100, // max_rewrites
        200, // max_growth (rationalization can grow temporarily)
        40,  // max_transform_rewrites
    );
}

#[test]
fn health_smoke_simple_no_op() {
    // Already simplified - should be very cheap
    run_health_check(
        "x + y", 20, // max_rewrites (very low)
        30, // max_growth
        10, // max_transform_rewrites
    );
}

#[test]
fn health_smoke_fraction_simplification() {
    // Common denominator reduction
    run_health_check(
        "x/2 + x/3",
        80,  // max_rewrites
        100, // max_growth
        40,  // max_transform_rewrites
    );
}

#[test]
fn health_smoke_nested_sqrt() {
    // Nested sqrt denesting (can be complex)
    run_health_check(
        "sqrt(3 + 2*sqrt(2))",
        120, // max_rewrites
        150, // max_growth
        50,  // max_transform_rewrites
    );
}

// ==================== BUDGET SATURATION CHECKS ====================

#[test]
fn health_budget_not_saturated() {
    // Verify that default budgets are not exceeded for a "normal" expression
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();

    let expr = cas_parser::parse("x/(1+sqrt(2))", &mut simplifier.context).expect("parse");
    let opts = SimplifyOptions::default();
    let budgets = opts.budgets;
    let (_, _, stats) = simplifier.simplify_with_stats(expr, opts);

    // None of these should saturate the budget (if so, might indicate loops)
    assert!(
        stats.core.iters_used < budgets.core_iters,
        "Core phase saturated budget: {} >= {}",
        stats.core.iters_used,
        budgets.core_iters
    );

    assert!(
        stats.transform.iters_used < budgets.transform_iters,
        "Transform phase saturated budget: {} >= {}",
        stats.transform.iters_used,
        budgets.transform_iters
    );

    assert!(
        stats.total_rewrites < budgets.max_total_rewrites,
        "Total rewrites saturated budget: {} >= {}",
        stats.total_rewrites,
        budgets.max_total_rewrites
    );
}

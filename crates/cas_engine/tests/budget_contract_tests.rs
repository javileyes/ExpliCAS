//! Contract tests for the unified Budget system.
//!
//! These tests validate that budget presets work correctly and
//! that the system fails fast without excessive allocation.

use cas_ast::Context;
use cas_engine::expand::expand_with_stats;
use cas_engine::{Budget, Metric, Operation, PassStats};
use cas_math::multinomial_expand::MultinomialExpandBudget;
use cas_parser::parse;

/// Test that expand pre-estimation catches explosive cases before materializing.
///
/// With a very low limit, the system should fail fast with:
/// - Almost no nodes_delta (nothing materialized)
/// - terms_materialized = 0 (precheck aborted)
#[test]
fn test_expand_fails_fast_without_allocation() {
    let mut ctx = Context::new();

    // (a+b)^50 would produce C(50+2-1, 2-1) = 51 terms
    // With MultinomialExpandBudget.max_output_terms = 10, it should abort
    let expr = parse("(a+b)^50", &mut ctx).unwrap();

    let nodes_before = ctx.stats().nodes_created;

    // Use very restrictive budget
    let budget = MultinomialExpandBudget {
        max_exp: 100,
        max_base_terms: 16,
        max_vars: 8,
        max_output_terms: 10, // Way below 51 terms needed
    };

    // The expand should return the original expression unexpanded
    // because precheck fails
    if let cas_ast::Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let result = cas_math::multinomial_expand::try_expand_multinomial_direct(
            &mut ctx, base, exp, &budget,
        );

        // Should return None (precheck failed, didn't expand)
        assert!(result.is_none(), "Expected precheck to abort expansion");
    }

    let nodes_after = ctx.stats().nodes_created;
    let nodes_delta = nodes_after - nodes_before;

    // Should have allocated almost nothing (just the parse, no expansion)
    assert!(
        nodes_delta < 10,
        "Expected fail-fast with minimal allocation, got {} nodes created",
        nodes_delta
    );
}

/// Test that expand_with_stats reports correct metrics.
#[test]
fn test_expand_with_stats_reports_metrics() {
    let mut ctx = Context::new();

    // (a+b)^3 → 4 terms: a³ + 3a²b + 3ab² + b³
    let expr = parse("(a+b)^3", &mut ctx).unwrap();

    let (result, stats) = expand_with_stats(&mut ctx, expr);

    // Should have expanded (not the same as input)
    assert_ne!(result, expr, "Expected expansion to occur");

    // Stats should report the operation
    assert_eq!(stats.op, Operation::Expand);

    // Should have created nodes
    assert!(stats.nodes_delta > 0, "Expected nodes_delta > 0");

    // Should estimate terms (binomial: 4 terms for (a+b)^3)
    assert!(
        stats.terms_materialized > 0,
        "Expected terms_materialized > 0, got {}",
        stats.terms_materialized
    );
}

/// Test that PassStats fields are correctly initialized.
#[test]
fn test_pass_stats_default() {
    let stats = PassStats::default();

    assert_eq!(stats.op, Operation::SimplifyCore); // Default operation
    assert_eq!(stats.rewrite_count, 0);
    assert_eq!(stats.nodes_delta, 0);
    assert_eq!(stats.terms_materialized, 0);
    assert_eq!(stats.poly_ops, 0);
    assert!(stats.stop_reason.is_none());
}

/// Test Budget preset_small has expected limits.
#[test]
fn test_budget_preset_small() {
    let budget = Budget::preset_small();

    // Check limits for SimplifyCore
    assert_eq!(
        budget.limit(Operation::SimplifyCore, Metric::RewriteSteps),
        5_000
    );
    assert_eq!(
        budget.limit(Operation::SimplifyCore, Metric::NodesCreated),
        25_000
    );
    assert_eq!(
        budget.limit(Operation::SimplifyCore, Metric::TermsMaterialized),
        10_000
    );
    assert_eq!(budget.limit(Operation::SimplifyCore, Metric::PolyOps), 500);

    // Same limits for Expand
    assert_eq!(
        budget.limit(Operation::Expand, Metric::TermsMaterialized),
        10_000
    );
}

/// Test Budget preset_cli has expected limits.
#[test]
fn test_budget_preset_cli() {
    let budget = Budget::preset_cli();

    assert_eq!(
        budget.limit(Operation::SimplifyCore, Metric::RewriteSteps),
        50_000
    );
    assert_eq!(
        budget.limit(Operation::Expand, Metric::TermsMaterialized),
        100_000
    );
}

/// Test Budget preset_unlimited has no limits.
#[test]
fn test_budget_preset_unlimited() {
    let budget = Budget::preset_unlimited();

    // All limits should be 0 (unlimited)
    assert_eq!(
        budget.limit(Operation::SimplifyCore, Metric::RewriteSteps),
        0
    );
    assert_eq!(
        budget.limit(Operation::Expand, Metric::TermsMaterialized),
        0
    );
}

/// Test Budget charge tracks usage correctly.
#[test]
fn test_budget_charge_tracks_usage() {
    let mut budget = Budget::preset_small();

    // Initially no usage
    assert_eq!(budget.used(Operation::Expand, Metric::TermsMaterialized), 0);

    // Charge some terms
    let result = budget.charge(Operation::Expand, Metric::TermsMaterialized, 100);
    assert!(result.is_ok());

    // Usage should be tracked
    assert_eq!(
        budget.used(Operation::Expand, Metric::TermsMaterialized),
        100
    );

    // Charge more
    let _ = budget.charge(Operation::Expand, Metric::TermsMaterialized, 200);
    assert_eq!(
        budget.used(Operation::Expand, Metric::TermsMaterialized),
        300
    );
}

/// Test Budget charge returns error when limit exceeded (strict mode).
#[test]
fn test_budget_charge_exceeds_limit() {
    let mut budget = Budget::preset_small();

    // preset_small has TermsMaterialized limit of 10,000
    // Try to charge more than the limit
    let result = budget.charge(Operation::Expand, Metric::TermsMaterialized, 15_000);

    assert!(
        result.is_err(),
        "Expected BudgetExceeded error in strict mode"
    );

    if let Err(exceeded) = result {
        assert_eq!(exceeded.op, Operation::Expand);
        assert_eq!(exceeded.metric, Metric::TermsMaterialized);
        assert_eq!(exceeded.limit, 10_000);
    }
}

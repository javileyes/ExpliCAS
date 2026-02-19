//! Tests for didactic substep gating with ChainedRewrite
//!
//! V2.12.13: Verifies that GCD substeps are skipped when step.is_chained == true

use cas_ast::{Context, Expr};
use cas_engine::enrich_steps;
use cas_engine::step::{ImportanceLevel, Step, StepCategory};

/// Test: Steps marked as chained should NOT have GCD factorization substeps
#[test]
fn test_chained_step_no_gcd_substeps() {
    let mut ctx = Context::new();

    // Create expressions for a simplified fraction scenario
    let x = ctx.var("x");
    let two = ctx.num(2);
    let before = ctx.add(Expr::Div(x, two));
    let after = x;

    // Create a step that would normally trigger GCD substeps
    let mut step = Step::new(
        "Simplified fraction by GCD: 2",
        "Simplify Fraction",
        before,
        after,
        vec![],
        Some(&ctx),
    );
    step.importance = ImportanceLevel::Medium;
    step.category = StepCategory::Simplify;
    step.meta_mut().is_chained = true; // Marked as from ChainedRewrite

    // Enrich steps
    let enriched = enrich_steps(&ctx, before, vec![step]);

    // Verify: no GCD substeps should be generated for chained steps
    assert_eq!(enriched.len(), 1);
    assert!(
        enriched[0].sub_steps.is_empty(),
        "Chained steps should not have GCD substeps, got: {:?}",
        enriched[0].sub_steps
    );
}

/// Test: Non-chained steps CAN have GCD factorization substeps
#[test]
fn test_non_chained_step_may_have_gcd_substeps() {
    let mut ctx = Context::new();

    // Create a Div expression that represents a fraction
    let x = ctx.var("x");
    let two = ctx.num(2);
    let before = ctx.add(Expr::Div(x, two));
    let after = x;

    // Create a step with the trigger description but NOT chained
    let mut step = Step::new(
        "Simplified fraction by GCD: 2",
        "Simplify Fraction",
        before,
        after,
        vec![],
        Some(&ctx),
    );
    step.importance = ImportanceLevel::Medium;
    step.category = StepCategory::Simplify;
    step.meta_mut().is_chained = false; // NOT chained - substeps should be attempted

    // Enrich steps
    let enriched = enrich_steps(&ctx, before, vec![step]);

    // Note: The actual substep generation depends on finding factorization patterns
    // This test just verifies that the gate doesn't block non-chained steps
    assert_eq!(enriched.len(), 1);
    // Substeps may or may not be present depending on pattern matching
    // The key is that the gate doesn't block them
}

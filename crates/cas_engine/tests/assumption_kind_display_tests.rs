//! Golden tests for AssumptionKind display filtering (V2.12.13).
//!
//! These tests verify that:
//! 1. DerivedFromRequires events are NOT displayed
//! 2. RequiresIntroduced events ARE displayed with ℹ️
//! 3. BranchChoice events ARE displayed with 🔀 and never promoted
//! 4. DomainExtension events ARE displayed with 🧿

use cas_ast::Context;
use cas_engine::assumptions::{AssumptionEvent, AssumptionKind};

// =============================================================================
// Test 1: DerivedFromRequires is hidden
// =============================================================================

/// Events classified as DerivedFromRequires should NOT be displayed.
/// This prevents redundant "x ≠ 0" messages when the input already implies it.
#[test]
fn derived_from_requires_not_displayed() {
    // Verify should_display returns false for DerivedFromRequires
    assert!(
        !AssumptionKind::DerivedFromRequires.should_display(),
        "DerivedFromRequires should NOT be displayed"
    );

    // Verify icon and label are empty
    assert_eq!(AssumptionKind::DerivedFromRequires.icon(), "");
    assert_eq!(AssumptionKind::DerivedFromRequires.label(), "Derived");
}

/// When a nonzero event has kind=DerivedFromRequires, it should be filtered out.
#[test]
fn nonzero_derived_from_requires_hidden() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    // Create event with default kind (DerivedFromRequires for nonzero)
    let event = AssumptionEvent::nonzero(&ctx, x);

    // Verify default kind is DerivedFromRequires
    assert_eq!(event.kind, AssumptionKind::DerivedFromRequires);

    // Should NOT be displayed
    assert!(
        !event.kind.should_display(),
        "nonzero with DerivedFromRequires kind should NOT display"
    );
}

// =============================================================================
// Test 2: RequiresIntroduced is shown (log rules)
// =============================================================================

/// Events classified as RequiresIntroduced SHOULD be displayed.
#[test]
fn requires_introduced_is_displayed() {
    assert!(
        AssumptionKind::RequiresIntroduced.should_display(),
        "RequiresIntroduced SHOULD be displayed"
    );

    // Verify icon is ℹ️
    assert_eq!(AssumptionKind::RequiresIntroduced.icon(), "ℹ️");
    assert_eq!(AssumptionKind::RequiresIntroduced.label(), "Requires");
}

/// positive() constructor defaults to RequiresIntroduced (for log rules).
#[test]
fn positive_defaults_to_requires_introduced() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let event = AssumptionEvent::positive(&ctx, x);

    // Should default to RequiresIntroduced
    assert_eq!(
        event.kind,
        AssumptionKind::RequiresIntroduced,
        "positive() should default to RequiresIntroduced for log rules"
    );

    // Message should be condition-only (no "Assumed" prefix)
    assert!(
        event.message.contains(">"),
        "message should contain '>' condition"
    );
    assert!(
        !event.message.starts_with("Assumed"),
        "message should NOT start with 'Assumed'"
    );

    // Should be displayed
    assert!(event.kind.should_display());
}

// =============================================================================
// Test 3: BranchChoice is shown and never promoted
// =============================================================================

/// BranchChoice events SHOULD be displayed with 🔀 icon.
#[test]
fn branch_choice_is_displayed() {
    assert!(
        AssumptionKind::BranchChoice.should_display(),
        "BranchChoice SHOULD be displayed"
    );

    assert_eq!(AssumptionKind::BranchChoice.icon(), "🔀");
    assert_eq!(AssumptionKind::BranchChoice.label(), "Branch");
}

/// inv_trig_principal_range defaults to BranchChoice.
#[test]
fn inv_trig_defaults_to_branch_choice() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let event = AssumptionEvent::inv_trig_principal_range(&ctx, "arcsin", x);

    assert_eq!(
        event.kind,
        AssumptionKind::BranchChoice,
        "inv_trig_principal_range should be BranchChoice"
    );

    // Verify message format
    assert!(
        event.message.contains("principal range"),
        "message should mention principal range"
    );

    // Should be displayed
    assert!(event.kind.should_display());
}

/// BranchChoice should never be promoted to RequiresIntroduced.
/// This is a contract test for the classifier.
#[test]
fn branch_choice_never_promoted() {
    use cas_engine::implicit_domain::{classify_assumption, DomainContext};

    let mut ctx = Context::new();
    let x = ctx.var("x");

    // Create a BranchChoice event
    let event = AssumptionEvent::inv_trig_principal_range(&ctx, "arcsin", x);
    assert_eq!(event.kind, AssumptionKind::BranchChoice);

    // Even with empty DomainContext (nothing implied), BranchChoice should stay as-is
    let dc = DomainContext::new(vec![]);
    let (new_kind, introduced) = classify_assumption(&ctx, &dc, &event);

    // BranchChoice MUST stay as BranchChoice
    assert_eq!(
        new_kind,
        AssumptionKind::BranchChoice,
        "BranchChoice must NEVER be promoted to RequiresIntroduced"
    );

    // Should NOT add to introduced requires
    assert!(
        introduced.is_none(),
        "BranchChoice should not add to introduced_requires"
    );
}

// =============================================================================
// Test 4: DomainExtension is shown
// =============================================================================

/// DomainExtension events SHOULD be displayed with 🧿 icon.
#[test]
fn domain_extension_is_displayed() {
    assert!(
        AssumptionKind::DomainExtension.should_display(),
        "DomainExtension SHOULD be displayed"
    );

    assert_eq!(AssumptionKind::DomainExtension.icon(), "🧿");
    assert_eq!(AssumptionKind::DomainExtension.label(), "Domain");
}

/// complex_principal_branch defaults to BranchChoice (ℂ principal branch is a branch choice).
#[test]
fn complex_principal_branch_is_branch_choice() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let event = AssumptionEvent::complex_principal_branch(&ctx, "sqrt", x);

    // Complex principal branch is classified as BranchChoice (choosing a branch)
    assert_eq!(
        event.kind,
        AssumptionKind::BranchChoice,
        "complex_principal_branch should be BranchChoice"
    );
}

// =============================================================================
// Test: HeuristicAssumption is shown
// =============================================================================

/// HeuristicAssumption events SHOULD be displayed with ⚠️ icon.
#[test]
fn heuristic_assumption_is_displayed() {
    assert!(
        AssumptionKind::HeuristicAssumption.should_display(),
        "HeuristicAssumption SHOULD be displayed"
    );

    assert_eq!(AssumptionKind::HeuristicAssumption.icon(), "⚠️");
    assert_eq!(AssumptionKind::HeuristicAssumption.label(), "Assumes");
}

// =============================================================================
// Test: Classifier reclassifies based on DomainContext
// =============================================================================

/// When a condition IS implied by global_requires, it should be reclassified
/// to DerivedFromRequires and not displayed.
#[test]
fn classifier_reclassifies_implied_to_derived() {
    use cas_engine::implicit_domain::{classify_assumption, DomainContext, ImplicitCondition};

    let mut ctx = Context::new();
    let x = ctx.var("x");

    // Create an event with RequiresIntroduced kind
    let event = AssumptionEvent::positive(&ctx, x);
    assert_eq!(event.kind, AssumptionKind::RequiresIntroduced);

    // Create DomainContext where x > 0 is already known (global)
    let dc = DomainContext::new(vec![ImplicitCondition::Positive(x)]);

    // Classify - should reclassify to DerivedFromRequires
    let (new_kind, introduced) = classify_assumption(&ctx, &dc, &event);

    assert_eq!(
        new_kind,
        AssumptionKind::DerivedFromRequires,
        "Implied condition should be DerivedFromRequires"
    );

    // Should NOT add to introduced (already known)
    assert!(
        introduced.is_none(),
        "Implied condition should not be added to introduced"
    );
}

/// When a condition is NOT implied, RequiresIntroduced should be promoted.
#[test]
fn classifier_promotes_unimplied_to_requires_introduced() {
    use cas_engine::implicit_domain::{classify_assumption, DomainContext, ImplicitCondition};

    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");

    // Create an event for x > 0
    let event = AssumptionEvent::positive(&ctx, x);

    // Create DomainContext where only y > 0 is known (x > 0 is NOT implied)
    let dc = DomainContext::new(vec![ImplicitCondition::Positive(y)]);

    // Classify
    let (new_kind, introduced) = classify_assumption(&ctx, &dc, &event);

    assert_eq!(
        new_kind,
        AssumptionKind::RequiresIntroduced,
        "Unimplied condition should be RequiresIntroduced"
    );

    // Should add to introduced requires
    assert!(
        introduced.is_some(),
        "Unimplied condition should be added to introduced"
    );
    assert!(matches!(
        introduced.unwrap(),
        ImplicitCondition::Positive(_)
    ));
}

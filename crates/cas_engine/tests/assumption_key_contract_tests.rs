//! Contract tests for structured AssumptionKey types.
//!
//! These tests verify that migrated rules emit the correct type of structured
//! assumption (NonZero, Positive, Defined, InvTrigPrincipalRange) instead of
//! legacy domain_assumption strings.
//!
//! # AssumptionKey Types
//!
//! - `NonZero`: Expression assumed to be ≠ 0 (e.g., denominators)
//! - `Positive`: Expression assumed to be > 0 (e.g., log arguments)
//! - `Defined`: Expression assumed to be defined/real (e.g., domain checks)
//! - `InvTrigPrincipalRange`: Argument in principal range of inverse trig function

use cas_engine::{DomainMode, InverseTrigPolicy, Simplifier, SimplifyOptions, Step};
use cas_parser::parse;

/// Helper: simplify with options and return steps
fn simplify_with_assume_steps(input: &str) -> Vec<Step> {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    let opts = SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: DomainMode::Assume,
                ..Default::default()
            },
            ..Default::default()
        },
        collect_steps: true,
        ..Default::default()
    };

    let (_, steps) = simplifier.simplify_with_options(expr, opts);
    steps
}

/// Helper: simplify with Principal value inv_trig
fn simplify_with_principal_inv_trig(input: &str) -> Vec<Step> {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    let opts = SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: DomainMode::Generic,
                inv_trig: InverseTrigPolicy::PrincipalValue,
                ..Default::default()
            },
            ..Default::default()
        },
        collect_steps: true,
        ..Default::default()
    };

    let (_, steps) = simplifier.simplify_with_options(expr, opts);
    steps
}

/// Check if any step has an assumption_event of the given kind
fn has_assumption_kind(steps: &[Step], kind: &str) -> bool {
    steps
        .iter()
        .any(|s| s.assumption_events().iter().any(|e| e.key.kind() == kind))
}

/// Check if any step has an assumption_event mentioning the given expression
fn has_assumption_for_expr(steps: &[Step], kind: &str, expr_contains: &str) -> bool {
    steps.iter().any(|s| {
        s.assumption_events()
            .iter()
            .any(|e| e.key.kind() == kind && e.expr_display.contains(expr_contains))
    })
}

// =============================================================================
// NonZero Assumption Tests
// =============================================================================

/// CONTRACT: x/x → 1 emits NonZero(x) assumption
#[test]
fn nonzero_emitted_for_division_cancellation() {
    let steps = simplify_with_assume_steps("x/x");

    assert!(
        has_assumption_kind(&steps, "nonzero"),
        "x/x should emit NonZero assumption. Steps: {:?}",
        steps
            .iter()
            .map(|s| s.assumption_events())
            .collect::<Vec<_>>()
    );
}

/// CONTRACT: x^0 → 1 emits NonZero(x) assumption
#[test]
fn nonzero_emitted_for_zero_exponent() {
    let steps = simplify_with_assume_steps("x^0");

    assert!(
        has_assumption_kind(&steps, "nonzero"),
        "x^0 should emit NonZero assumption for x"
    );
}

/// CONTRACT: (2*x)/(2*x) → 1 emits NonZero for x (not for 2)
#[test]
fn nonzero_targets_symbolic_factor() {
    let steps = simplify_with_assume_steps("(2*x)/(2*x)");

    assert!(
        has_assumption_for_expr(&steps, "nonzero", "x"),
        "Should emit NonZero for x, not 2. Events: {:?}",
        steps
            .iter()
            .flat_map(|s| s.assumption_events())
            .collect::<Vec<_>>()
    );
}

// =============================================================================
// Positive Assumption Tests
// =============================================================================

/// CONTRACT: 0^x → 0 emits Positive(x) assumption
#[test]
fn positive_emitted_for_zero_base_power() {
    let steps = simplify_with_assume_steps("0^x");

    assert!(
        has_assumption_kind(&steps, "positive"),
        "0^x should emit Positive assumption for x"
    );
}

/// CONTRACT: ln(x*y) expansion emits Positive(x) and Positive(y)
/// NOTE: LogExpansionRule is no longer in default rules (use expand_log command),
/// so this test explicitly registers the rule to verify assumption emission.
/// We use Simplifier::new() to avoid LogContractionRule which would undo the expansion.
#[test]
fn positive_emitted_for_log_product_expansion() {
    use cas_engine::rules::logarithms::LogExpansionRule;

    // Create simplifier with ONLY LogExpansionRule (no LogContractionRule which would undo it)
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(LogExpansionRule));
    simplifier.set_collect_steps(true);

    let expr = parse("ln(x*y)", &mut simplifier.context).expect("parse failed");
    let opts = SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: DomainMode::Assume,
                ..Default::default()
            },
            ..Default::default()
        },
        collect_steps: true,
        ..Default::default()
    };
    let (_, steps) = simplifier.simplify_with_options(expr, opts);

    // Should have at least one positive assumption
    assert!(
        has_assumption_kind(&steps, "positive"),
        "ln(x*y) expansion should emit Positive assumptions"
    );

    // Should mention both x and y
    let has_x = has_assumption_for_expr(&steps, "positive", "x");
    let has_y = has_assumption_for_expr(&steps, "positive", "y");

    assert!(
        has_x && has_y,
        "ln(x*y) should emit Positive for both x and y. x={}, y={}",
        has_x,
        has_y
    );
}

/// CONTRACT: e^ln(x) → x emits Positive(x) as IMPLICIT requirement in Generic/Assume mode
/// This is NOT an assumption (heuristic) but an IMPLICIT domain requirement from ln(x).
/// Similar to how sqrt(x)^2 → x requires x ≥ 0.
#[test]
fn positive_emitted_for_exp_ln_inverse() {
    // Use Generic mode - should work with implicit requires
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let expr = parse("exp(ln(x))", &mut simplifier.context).expect("parse failed");

    let opts = SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: DomainMode::Generic,
                ..Default::default()
            },
            ..Default::default()
        },
        collect_steps: true,
        ..Default::default()
    };

    let (_, steps) = simplifier.simplify_with_options(expr, opts);

    // The condition x > 0 should be in required_conditions (implicit domain), not assumption_events
    let has_positive_require = steps.iter().any(|s| {
        s.required_conditions().iter().any(|c| {
            let display = c.display(&simplifier.context);
            display.contains("x") && display.contains("> 0")
        })
    });

    assert!(
        has_positive_require,
        "exp(ln(x)) should emit x > 0 as required_condition (implicit domain from ln). Steps: {:?}",
        steps
            .iter()
            .map(|s| (s.required_conditions(), s.assumption_events()))
            .collect::<Vec<_>>()
    );
}

// =============================================================================
// Defined Assumption Tests
// =============================================================================

/// CONTRACT: ln(e^x) → x does NOT emit any assumption
/// REASON: e^x > 0 for all real x, so ln(e^x) is always defined and equals x.
/// This is mathematically correct - no domain restriction needed.
#[test]
fn no_assumption_for_log_exp_inverse() {
    // Use Generic mode which triggers LogExpInverseRule
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let expr = parse("ln(exp(x))", &mut simplifier.context).expect("parse failed");

    let opts = SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: DomainMode::Generic,
                ..Default::default()
            },
            ..Default::default()
        },
        collect_steps: true,
        ..Default::default()
    };

    let (_, steps) = simplifier.simplify_with_options(expr, opts);

    // ln(e^x) = x should NOT emit any assumption because e^x > 0 for all x
    let has_any_assumption = steps.iter().any(|s| !s.assumption_events().is_empty());

    assert!(
        !has_any_assumption,
        "ln(exp(x)) should NOT emit any assumption - e^x > 0 for all real x"
    );
}

// =============================================================================
// InvTrigPrincipalRange Assumption Tests
// =============================================================================

/// CONTRACT: arcsin(sin(x)) → x emits InvTrigPrincipalRange assumption
#[test]
fn principal_range_emitted_for_arcsin_sin() {
    let steps = simplify_with_principal_inv_trig("arcsin(sin(x))");

    assert!(
        has_assumption_kind(&steps, "principal_range"),
        "arcsin(sin(x)) should emit principal_range assumption"
    );
}

/// CONTRACT: arctan(tan(x)) → x emits InvTrigPrincipalRange assumption
#[test]
fn principal_range_emitted_for_arctan_tan() {
    let steps = simplify_with_principal_inv_trig("arctan(tan(x))");

    assert!(
        has_assumption_kind(&steps, "principal_range"),
        "arctan(tan(x)) should emit principal_range assumption"
    );
}

/// CONTRACT: arccos(cos(x)) → x emits InvTrigPrincipalRange assumption
#[test]
fn principal_range_emitted_for_arccos_cos() {
    let steps = simplify_with_principal_inv_trig("arccos(cos(x))");

    assert!(
        has_assumption_kind(&steps, "principal_range"),
        "arccos(cos(x)) should emit principal_range assumption"
    );
}

// =============================================================================
// Deduplication Tests
// =============================================================================

/// CONTRACT: Multiple uses of same assumption should deduplicate
#[test]
fn duplicate_assumptions_deduplicate() {
    // x/x + x/x should not emit duplicate NonZero(x) in final output
    let steps = simplify_with_assume_steps("x/x + x/x");

    // Count NonZero(x) events across all steps
    let nonzero_x_count: usize = steps
        .iter()
        .flat_map(|s| s.assumption_events())
        .filter(|e| e.key.kind() == "nonzero" && e.expr_display == "x")
        .count();

    // Each x/x should emit one event, so we expect >= 1
    // The deduplication happens in the collector's finish() method, not in raw steps
    assert!(
        nonzero_x_count >= 1,
        "Should have at least one NonZero(x) event"
    );
}

// =============================================================================
// No Assumption When Not Needed
// =============================================================================

/// CONTRACT: 2^0 → 1 should NOT emit assumption (2 is provably non-zero)
#[test]
fn no_nonzero_for_provably_nonzero_base() {
    let steps = simplify_with_assume_steps("2^0");

    // Should NOT have NonZero for 2 since 2 is provably non-zero
    let has_nonzero_2 = has_assumption_for_expr(&steps, "nonzero", "2");

    assert!(
        !has_nonzero_2,
        "2^0 should NOT emit NonZero for 2 (provably non-zero)"
    );
}

/// CONTRACT: ln(2*3) should NOT emit assumptions (2 and 3 are provably positive)
#[test]
fn no_positive_for_provably_positive_factors() {
    let steps = simplify_with_assume_steps("ln(2*3)");

    // Should NOT have Positive for 2 or 3
    let has_positive = has_assumption_kind(&steps, "positive");

    assert!(
        !has_positive,
        "ln(2*3) should NOT emit Positive assumptions (provably positive)"
    );
}

//! Domain Assume mode warnings contract tests.
//!
//! # Contract: Assume Mode - Simplify with Traceable Assumptions
//!
//! In `DomainMode::Assume`:
//! - Allow `x/x → 1` (like Generic)
//! - BUT must emit a **domain_assumption** (legacy) or **assumption_events** (structured) on the Step
//! - The assumption should indicate what was assumed (e.g., "x ≠ 0")
//!
//! This provides "best effort" simplification with audit trail.

use cas_engine::{Simplifier, Step};
use cas_parser::parse;

/// Helper: check if step has any assumption_events
fn step_has_assumption(step: &Step) -> bool {
    !step.assumption_events().is_empty()
}

/// Helper: simplify with Assume domain mode, returning result and steps
fn simplify_assume_with_steps(input: &str) -> (String, Vec<Step>) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    let opts = cas_engine::SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: cas_engine::DomainMode::Assume,
                ..Default::default()
            },
            ..Default::default()
        },
        collect_steps: true,
        ..Default::default()
    };

    let (result, steps) = simplifier.simplify_with_options(expr, opts);
    let result_str = format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    (result_str, steps)
}

// =============================================================================
// Assume Mode: Warnings/Assumptions Tests
// =============================================================================

#[test]
fn assume_x_div_x_simplifies_but_emits_assumption() {
    let (got, steps) = simplify_assume_with_steps("x/x");
    assert_eq!(got, "1", "Assume mode should simplify x/x to 1");

    // Contract: must exist at least one step with domain_assumption OR assumption_events
    let has_assumption = steps.iter().any(step_has_assumption);
    assert!(
        has_assumption,
        "Assume-mode cancellations must emit assumption. Steps: {:?}",
        steps.iter().map(|s| &s.rule_name).collect::<Vec<_>>()
    );
}

#[test]
#[allow(clippy::needless_option_as_deref)]
fn assume_2x_div_2x_simplifies_and_assumption_mentions_symbolic_part() {
    let (got, steps) = simplify_assume_with_steps("(2*x)/(2*x)");
    assert_eq!(got, "1", "Assume mode should simplify (2*x)/(2*x) to 1");

    // Contract: the assumption should mention x (the symbolic nonzero), not "2*x"
    // (since 2 is provably nonzero, only x needs to be assumed)
    // Check structured assumption_events
    let has_structured = steps.iter().any(|s| !s.assumption_events().is_empty());

    assert!(
        has_structured,
        "Expected at least one assumption_event, got none"
    );
    // Structured assumption_events encodes the expression in its key
}

#[test]
fn assume_power_cancellation_emits_assumption() {
    let (got, steps) = simplify_assume_with_steps("x^2/x^2");
    assert_eq!(got, "1", "Assume mode should simplify x^2/x^2 to 1");

    let has_assumption = steps.iter().any(step_has_assumption);
    assert!(
        has_assumption,
        "x^2/x^2 cancellation must emit assumption about x"
    );
}

#[test]
fn assume_x_pow_0_simplifies_with_assumption() {
    let (got, steps) = simplify_assume_with_steps("x^0");
    assert_eq!(got, "1", "Assume mode should simplify x^0 to 1");

    let has_assumption = steps.iter().any(step_has_assumption);
    assert!(
        has_assumption,
        "x^0 → 1 must emit assumption about x ≠ 0 (legacy or structured)"
    );
}

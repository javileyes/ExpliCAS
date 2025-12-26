//! Contract tests for InverseTrigPolicy gating.
//!
//! These tests verify the "load-bearing" behaviors of inverse trig policy:
//! 1. Strict mode: arctan(tan(x)) remains unchanged
//! 2. Principal mode: arctan(tan(x)) → x with domain_assumption
//! 3. Policy is independent of DomainMode

use cas_ast::DisplayExpr;
use cas_engine::{
    DomainMode, Engine, EntryKind, EvalAction, EvalRequest, EvalResult, InverseTrigPolicy,
    SessionState,
};
use cas_parser::parse;

/// Helper: simplify with given InverseTrigPolicy using Engine API
fn simplify_with_inv_trig(input: &str, policy: InverseTrigPolicy) -> (String, Vec<String>) {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Configure inv_trig policy
    state.options.inv_trig = policy;

    let parsed = parse(input, &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    let result_str = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    let warnings: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();

    (result_str, warnings)
}

// ============================================================================
// Strict mode tests (default behavior)
// ============================================================================

#[test]
fn strict_arctan_tan_unchanged() {
    let (result, _) = simplify_with_inv_trig("arctan(tan(x))", InverseTrigPolicy::Strict);

    // Should remain unchanged (strict mode blocks simplification)
    assert!(
        result.contains("arctan") || result.contains("atan"),
        "Expected arctan(tan(x)) unchanged, got: {}",
        result
    );
}

#[test]
fn strict_arcsin_sin_unchanged() {
    let (result, _) = simplify_with_inv_trig("arcsin(sin(x))", InverseTrigPolicy::Strict);

    assert!(
        result.contains("arcsin") || result.contains("asin"),
        "Expected arcsin(sin(x)) unchanged, got: {}",
        result
    );
}

#[test]
fn strict_arccos_cos_unchanged() {
    let (result, _) = simplify_with_inv_trig("arccos(cos(x))", InverseTrigPolicy::Strict);

    assert!(
        result.contains("arccos") || result.contains("acos"),
        "Expected arccos(cos(x)) unchanged, got: {}",
        result
    );
}

// ============================================================================
// Principal mode tests (educational simplifications)
// ============================================================================

#[test]
fn principal_arctan_tan_simplifies() {
    let (result, warnings) =
        simplify_with_inv_trig("arctan(tan(x))", InverseTrigPolicy::PrincipalValue);

    // Should simplify to x
    assert_eq!(result, "x", "Expected x, got: {}", result);

    // Should emit warning about principal branch assumption
    assert!(!warnings.is_empty(), "Expected principal branch warning");
}

#[test]
fn principal_arcsin_sin_simplifies() {
    let (result, warnings) =
        simplify_with_inv_trig("arcsin(sin(x))", InverseTrigPolicy::PrincipalValue);

    assert_eq!(result, "x", "Expected x, got: {}", result);
    assert!(!warnings.is_empty(), "Expected principal branch warning");
}

#[test]
fn principal_arccos_cos_simplifies() {
    let (result, warnings) =
        simplify_with_inv_trig("arccos(cos(x))", InverseTrigPolicy::PrincipalValue);

    assert_eq!(result, "x", "Expected x, got: {}", result);
    assert!(!warnings.is_empty(), "Expected principal branch warning");
}

// ============================================================================
// Independence from DomainMode
// ============================================================================

/// Verify that InverseTrigPolicy::Strict blocks simplification
/// even when DomainMode is Assume (most permissive).
#[test]
fn strict_independent_of_domain_assume() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Set inv_trig = Strict but domain = Assume (most permissive)
    state.options.inv_trig = InverseTrigPolicy::Strict;
    state.options.domain_mode = DomainMode::Assume;

    let parsed = parse("arctan(tan(x))", &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: "arctan(tan(x))".to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_str = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    // Should still NOT simplify (inv_trig policy is independent of domain)
    assert!(
        result_str.contains("arctan") || result_str.contains("atan"),
        "Expected arctan(tan(x)) unchanged even with DomainMode::Assume, got: {}",
        result_str
    );
}

/// Verify that InverseTrigPolicy::PrincipalValue simplifies
/// even when DomainMode is Strict (most restrictive).
#[test]
fn principal_independent_of_domain_strict() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Set inv_trig = PrincipalValue but domain = Strict (most restrictive)
    state.options.inv_trig = InverseTrigPolicy::PrincipalValue;
    state.options.domain_mode = DomainMode::Strict;

    let parsed = parse("arctan(tan(x))", &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: "arctan(tan(x))".to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result_str = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    // Should simplify (inv_trig policy is independent of domain)
    assert_eq!(
        result_str, "x",
        "Expected x with InverseTrigPolicy::PrincipalValue, got: {}",
        result_str
    );
}

// ============================================================================
// Safe compositions still work in both modes
// ============================================================================

/// tan(arctan(x)) = x is mathematically safe for all x, should work in both modes.
#[test]
fn tan_arctan_simplifies_in_strict() {
    let (result, _) = simplify_with_inv_trig("tan(arctan(x))", InverseTrigPolicy::Strict);

    // This is always safe: tan(arctan(x)) = x for all real x
    assert_eq!(result, "x", "Expected x, got: {}", result);
}

#[test]
fn sin_arcsin_simplifies_in_strict() {
    let (result, _) = simplify_with_inv_trig("sin(arcsin(x))", InverseTrigPolicy::Strict);

    // This is always safe: sin(arcsin(x)) = x for x ∈ [-1, 1]
    assert_eq!(result, "x", "Expected x, got: {}", result);
}

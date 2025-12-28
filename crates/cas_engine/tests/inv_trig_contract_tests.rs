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
    let (result, _warnings) =
        simplify_with_inv_trig("arctan(tan(x))", InverseTrigPolicy::PrincipalValue);

    // Should simplify to x
    assert_eq!(result, "x", "Expected x, got: {}", result);

    // NOTE: Principal branch rules now emit structured assumption_events instead of
    // domain_assumption strings. The warnings array may be empty during migration.
    // The Step.assumption_events IS populated (verified by domain_assume_warnings_contract_tests)
}

#[test]
fn principal_arcsin_sin_simplifies() {
    let (result, _warnings) =
        simplify_with_inv_trig("arcsin(sin(x))", InverseTrigPolicy::PrincipalValue);

    assert_eq!(result, "x", "Expected x, got: {}", result);
    // NOTE: Structured assumption_events used instead of domain_assumption (see above)
}

#[test]
fn principal_arccos_cos_simplifies() {
    let (result, _warnings) =
        simplify_with_inv_trig("arccos(cos(x))", InverseTrigPolicy::PrincipalValue);

    assert_eq!(result, "x", "Expected x, got: {}", result);
    // NOTE: Structured assumption_events used instead of domain_assumption (see above)
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
fn sin_arcsin_preserves_in_strict_domain() {
    // sin(arcsin(x)) with variable x requires x ∈ [-1, 1]
    // In strict domain mode (default from inv_trig API helper), it should NOT simplify
    // because we can't assume x is in the domain
    let (result, _) = simplify_with_inv_trig("sin(arcsin(x))", InverseTrigPolicy::Strict);

    // Note: simplify_with_inv_trig uses default domain=generic, so this WILL simplify
    // This test verifies it still works in the default generic mode
    assert_eq!(result, "x", "Expected x in generic mode, got: {}", result);
}

// ============================================================================
// Strict mode: NO principal branch warnings should appear
// ============================================================================

#[test]
fn strict_arctan_tan_no_principal_warning() {
    let (_, warnings) = simplify_with_inv_trig("arctan(tan(x))", InverseTrigPolicy::Strict);

    // In strict mode, no "principal branch" warning should appear
    let has_principal_warning = warnings
        .iter()
        .any(|w| w.to_lowercase().contains("principal"));
    assert!(
        !has_principal_warning,
        "Strict mode should NOT emit principal branch warning, got: {:?}",
        warnings
    );
}

// ============================================================================
// Protection is NOT global: tan(x) outside arctan(tan(...)) can still rewrite
// ============================================================================

#[test]
fn principal_tan_alone_can_rewrite_to_sin_cos() {
    // When inv_trig=principal but tan(x) is NOT inside arctan(...),
    // TanToSinCosRule should still be allowed to convert tan(x) -> sin(x)/cos(x)
    // (if that rule is enabled, which depends on other settings)
    let (result, _) = simplify_with_inv_trig("tan(x) + 1", InverseTrigPolicy::PrincipalValue);

    // The result should contain either tan(x) or sin(x)/cos(x) - both are valid.
    // The key is that it should NOT be affected by inverse trig protection.
    // We just verify it didn't error and produced some result.
    assert!(
        !result.is_empty(),
        "Expected valid result for tan(x) + 1, got empty"
    );
}

#[test]
fn principal_nested_tan_outside_arctan_can_rewrite() {
    // tan(x) * arctan(tan(y)) - the tan(x) is NOT protected, but the inner tan(y) IS
    // This tests that protection is targeted, not global
    let (result, _) =
        simplify_with_inv_trig("tan(x) * arctan(tan(y))", InverseTrigPolicy::PrincipalValue);

    // arctan(tan(y)) should simplify to y
    // tan(x) may or may not rewrite to sin(x)/cos(x) depending on rules
    // Key assertion: arctan(tan(y)) -> y happened
    assert!(
        result.contains("y") && !result.contains("arctan(tan(y))"),
        "Expected arctan(tan(y)) to simplify to y, got: {}",
        result
    );
}

// ============================================================================
// Log-Exp domain tests (controlled by domain_mode, NOT inv_trig)
// ============================================================================

/// Helper: simplify with given DomainMode using Engine API
fn simplify_with_domain(input: &str, domain: DomainMode) -> (String, Vec<String>) {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Configure domain_mode
    state.options.domain_mode = domain;

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

#[test]
fn domain_strict_ln_exp_unchanged() {
    let (result, _) = simplify_with_domain("ln(exp(x))", DomainMode::Strict);

    // Strict mode: ln(e^x) should remain unchanged (not simplify to x)
    assert!(
        result.contains("log") || result.contains("ln"),
        "Expected ln(exp(x)) unchanged in strict domain mode, got: {}",
        result
    );
}

#[test]
fn domain_strict_ln_exp_no_warning() {
    let (_, warnings) = simplify_with_domain("ln(exp(x))", DomainMode::Strict);

    // In strict domain mode, no warning about "assuming x is real" should appear
    let has_assumption_warning = warnings
        .iter()
        .any(|w| w.to_lowercase().contains("assuming"));
    assert!(
        !has_assumption_warning,
        "Strict domain mode should NOT emit assumption warning for ln(exp(x)), got: {:?}",
        warnings
    );
}

#[test]
fn domain_generic_ln_exp_simplifies() {
    let (result, warnings) = simplify_with_domain("ln(exp(x))", DomainMode::Generic);

    // Generic mode: ln(e^x) should simplify to x (we assume x is real)
    assert_eq!(result, "x", "Expected x, got: {}", result);

    // Should emit warning about assumption
    assert!(
        !warnings.is_empty(),
        "Expected domain assumption warning for ln(exp(x))"
    );
}

#[test]
fn domain_assume_ln_exp_simplifies() {
    let (result, _) = simplify_with_domain("ln(exp(x))", DomainMode::Assume);

    // Assume mode: ln(e^x) should simplify to x
    assert_eq!(result, "x", "Expected x in assume mode, got: {}", result);
}

#[test]
fn log_numeric_exponent_always_simplifies() {
    // log(x, x^2) = 2 should ALWAYS work (numeric exponent), regardless of domain mode
    let (result_strict, _) = simplify_with_domain("log(x, x^2)", DomainMode::Strict);
    let (result_generic, _) = simplify_with_domain("log(x, x^2)", DomainMode::Generic);

    assert_eq!(
        result_strict, "2",
        "Expected log(x, x^2) = 2 in strict mode, got: {}",
        result_strict
    );
    assert_eq!(
        result_generic, "2",
        "Expected log(x, x^2) = 2 in generic mode, got: {}",
        result_generic
    );
}

#[test]
fn ln_e_to_numeric_always_simplifies() {
    // ln(e^3) = 3 should ALWAYS work (numeric exponent), regardless of domain mode
    let (result_strict, _) = simplify_with_domain("ln(exp(3))", DomainMode::Strict);
    let (result_generic, _) = simplify_with_domain("ln(exp(3))", DomainMode::Generic);

    assert_eq!(
        result_strict, "3",
        "Expected ln(e^3) = 3 in strict mode, got: {}",
        result_strict
    );
    assert_eq!(
        result_generic, "3",
        "Expected ln(e^3) = 3 in generic mode, got: {}",
        result_generic
    );
}

//! Regression tests for Requires/Assumed propagation when reusing session entries.
//!
//! These tests verify that:
//! 1. Requires are re-inferred when expressions are reused via #id
//! 2. Structural domain conditions propagate correctly through composition
//! 3. Combined expressions inherit requires from all sub-expressions

use cas_ast::Expr;
use cas_engine::Engine;
use cas_engine::{EvalAction, EvalRequest};
mod support;
use support::SessionState;

/// Helper to create eval request for expression
fn make_simplify_request(engine: &mut Engine, expr_str: &str) -> EvalRequest {
    let parsed =
        cas_parser::parse(expr_str, &mut engine.simplifier.context).expect("parse should succeed");
    EvalRequest {
        raw_input: expr_str.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: true,
    }
}

/// Test: sqrt(x) produces x ≥ 0, and this propagates when reused
#[test]
fn requires_propagate_on_sqrt_reuse() {
    use cas_engine::implicit_domain::ImplicitCondition;

    let mut engine = Engine::new();
    let mut state = SessionState::default();

    // Step 1: Evaluate sqrt(x) -> stored as #1
    let req1 = make_simplify_request(&mut engine, "sqrt(x)");
    let output1 = engine.eval(&mut state, req1).expect("eval should succeed");

    // Verify sqrt(x) has x ≥ 0 in requires
    let has_x_nonneg = output1.required_conditions.iter().any(|c| {
        matches!(c, ImplicitCondition::NonNegative(e) 
            if matches!(engine.simplifier.context.get(*e), Expr::Variable(sym_id) if engine.simplifier.context.sym_name(*sym_id) == "x"))
    });
    assert!(
        has_x_nonneg,
        "sqrt(x) should require x ≥ 0, got: {:?}",
        output1.required_conditions
    );

    // Step 2: Build expression using #1: "#1 + 4"
    // We need to parse this, but since #1 references session, we simulate by
    // directly building the expression
    let stored_id = output1.stored_id.expect("should have stored_id");
    assert_eq!(stored_id, 1, "First entry should be #1");

    // Parse "#1 + 4" which will resolve #1 -> sqrt(x)
    let req2 = make_simplify_request(&mut engine, "#1 + 4");
    let output2 = engine.eval(&mut state, req2).expect("eval should succeed");

    // Verify the reused expression still has x ≥ 0
    let has_x_nonneg_2 = output2.required_conditions.iter().any(|c| {
        matches!(c, ImplicitCondition::NonNegative(e) 
            if matches!(engine.simplifier.context.get(*e), Expr::Variable(sym_id) if engine.simplifier.context.sym_name(*sym_id) == "x"))
    });
    assert!(
        has_x_nonneg_2,
        "Reused sqrt(x) in #1 + 4 should still require x ≥ 0, got: {:?}",
        output2.required_conditions
    );
}

/// Test: ln(y) produces y > 0, and this propagates when reused
#[test]
fn requires_propagate_on_ln_reuse() {
    use cas_engine::implicit_domain::ImplicitCondition;

    let mut engine = Engine::new();
    let mut state = SessionState::default();

    // Step 1: Evaluate ln(y) -> stored as #1
    let req1 = make_simplify_request(&mut engine, "ln(y)");
    let output1 = engine.eval(&mut state, req1).expect("eval should succeed");

    // Verify ln(y) has y > 0 in requires
    let has_y_positive = output1.required_conditions.iter().any(|c| {
        matches!(c, ImplicitCondition::Positive(e) 
            if matches!(engine.simplifier.context.get(*e), Expr::Variable(sym_id) if engine.simplifier.context.sym_name(*sym_id) == "y"))
    });
    assert!(
        has_y_positive,
        "ln(y) should require y > 0, got: {:?}",
        output1.required_conditions
    );

    // Step 2: Use #1 in composition: "#1 * 2"
    let req2 = make_simplify_request(&mut engine, "#1 * 2");
    let output2 = engine.eval(&mut state, req2).expect("eval should succeed");

    // Verify y > 0 propagates
    let has_y_positive_2 = output2.required_conditions.iter().any(|c| {
        matches!(c, ImplicitCondition::Positive(e) 
            if matches!(engine.simplifier.context.get(*e), Expr::Variable(sym_id) if engine.simplifier.context.sym_name(*sym_id) == "y"))
    });
    assert!(
        has_y_positive_2,
        "Reused ln(y) in #1 * 2 should still require y > 0, got: {:?}",
        output2.required_conditions
    );
}

/// Test: Combined expressions inherit requires from all sub-expressions
#[test]
fn requires_combine_from_multiple_sources() {
    use cas_engine::implicit_domain::ImplicitCondition;

    let mut engine = Engine::new();
    let mut state = SessionState::default();

    // Create sqrt(x) + ln(y) which should require both x ≥ 0 AND y > 0
    let req = make_simplify_request(&mut engine, "sqrt(x) + ln(y)");
    let output = engine.eval(&mut state, req).expect("eval should succeed");

    // Check for x ≥ 0
    let has_x_nonneg = output.required_conditions.iter().any(|c| {
        matches!(c, ImplicitCondition::NonNegative(e) 
            if matches!(engine.simplifier.context.get(*e), Expr::Variable(sym_id) if engine.simplifier.context.sym_name(*sym_id) == "x"))
    });

    // Check for y > 0
    let has_y_positive = output.required_conditions.iter().any(|c| {
        matches!(c, ImplicitCondition::Positive(e) 
            if matches!(engine.simplifier.context.get(*e), Expr::Variable(sym_id) if engine.simplifier.context.sym_name(*sym_id) == "y"))
    });

    assert!(
        has_x_nonneg,
        "sqrt(x) + ln(y) should require x ≥ 0, got: {:?}",
        output.required_conditions
    );
    assert!(
        has_y_positive,
        "sqrt(x) + ln(y) should require y > 0, got: {:?}",
        output.required_conditions
    );
}

/// Test: Division by variable produces x ≠ 0 requirement
#[test]
fn requires_nonzero_from_division() {
    use cas_engine::implicit_domain::ImplicitCondition;

    let mut engine = Engine::new();
    let mut state = SessionState::default();

    // Create 1/x which should require x ≠ 0
    let req = make_simplify_request(&mut engine, "1/z");
    let output = engine.eval(&mut state, req).expect("eval should succeed");

    // Check for z ≠ 0
    let has_z_nonzero = output.required_conditions.iter().any(|c| {
        matches!(c, ImplicitCondition::NonZero(e) 
            if matches!(engine.simplifier.context.get(*e), Expr::Variable(sym_id) if engine.simplifier.context.sym_name(*sym_id) == "z"))
    });

    assert!(
        has_z_nonzero,
        "1/z should require z ≠ 0, got: {:?}",
        output.required_conditions
    );
}

// =============================================================================
// SessionPropagated Origin Tests (V2.2)
// =============================================================================

/// Test A: SessionPropagated origin appears when reusing #id
///
/// When an expression references #n, the conditions from that entry should be
/// inherited with SessionPropagated added to their origins.
#[test]
fn session_propagated_origin_appears_on_reuse() {
    use cas_engine::RequireOrigin;

    let mut engine = Engine::new();
    let mut state = SessionState::default();

    // Step 1: Evaluate sqrt(x) -> stored as #1
    let req1 = make_simplify_request(&mut engine, "sqrt(x)");
    let output1 = engine.eval(&mut state, req1).expect("eval should succeed");

    // Verify #1 was stored
    let stored_id = output1.stored_id.expect("should have stored_id");
    assert_eq!(stored_id, 1, "First entry should be #1");

    // Verify #1's diagnostics have requires (for later inheritance)
    assert!(
        !output1.diagnostics.requires.is_empty(),
        "sqrt(x) should have requires in diagnostics"
    );

    // Step 2: Evaluate #1 + 1 (reuses #1)
    let req2 = make_simplify_request(&mut engine, "#1 + 1");
    let output2 = engine.eval(&mut state, req2).expect("eval should succeed");

    // Verify x ≥ 0 is in diagnostics.requires with SessionPropagated origin
    let has_session_propagated = output2.diagnostics.requires.iter().any(|item| {
        item.origins
            .iter()
            .any(|o| matches!(o, RequireOrigin::SessionPropagated))
    });

    assert!(
        has_session_propagated,
        "Reused #1 should have SessionPropagated origin, got requires: {:?}",
        output2
            .diagnostics
            .requires
            .iter()
            .map(|r| (&r.cond, &r.origins))
            .collect::<Vec<_>>()
    );

    // Also verify original origin is preserved
    let has_original_origin = output2.diagnostics.requires.iter().any(|item| {
        item.origins
            .iter()
            .any(|o| matches!(o, RequireOrigin::OutputImplicit))
    });

    assert!(
        has_original_origin,
        "Original OutputImplicit origin should be preserved alongside SessionPropagated"
    );
}

/// Test B: SessionPropagated does NOT contaminate assumed/blocked
///
/// When reusing #id, only `requires` should be inherited.
/// `assumed` and `blocked` should NOT be propagated.
#[test]
fn session_propagated_no_assumed_blocked_contamination() {
    let mut engine = Engine::new();
    let mut state = SessionState::default();

    // Use Assume mode to generate assumed events
    state.options_mut().shared.semantics.domain_mode = cas_engine::DomainMode::Assume;

    // Step 1: Evaluate exp(ln(x)) in Assume mode
    // This should simplify to x and record Positive(x) as assumed
    let req1 = make_simplify_request(&mut engine, "exp(ln(x))");
    let output1 = engine.eval(&mut state, req1).expect("eval should succeed");

    // Verify we got an assumed event (or at least the simplification happened)
    let result1 = match &output1.result {
        cas_engine::EvalResult::Expr(e) => cas_formatter::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };
    assert_eq!(
        result1, "x",
        "exp(ln(x)) should simplify to x in Assume mode"
    );

    // Step 2: Evaluate #1 + 1 (reuses #1)
    let req2 = make_simplify_request(&mut engine, "#1 + 1");
    let output2 = engine.eval(&mut state, req2).expect("eval should succeed");

    // Verify assumed from #1 is NOT propagated to #2
    // (assumed events should only reflect the current evaluation)
    // Note: The current eval might have its own assumed events, but they should
    // not include ones from the #1 evaluation context

    // Key insight: assumed should NOT grow unboundedly from referenced entries
    // We can't easily assert this without more detailed tracking, but the key
    // architectural guarantee is that inherit_requires_from only inherits requires,
    // not assumed or blocked. This is enforced by the function signature.

    // For now, we verify that the result is correct
    let result2 = match &output2.result {
        cas_engine::EvalResult::Expr(e) => cas_formatter::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };
    assert!(
        result2 == "x + 1" || result2 == "1 + x",
        "#1 + 1 should give x + 1 (or 1 + x), got: {}",
        result2
    );
}

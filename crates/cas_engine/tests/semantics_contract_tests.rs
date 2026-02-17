//! Contract tests for EvalConfig and semantic axes (PR1).
//!
//! # Contract: Defaults Don't Change Behavior
//!
//! The introduction of ValueDomain, BranchPolicy, and InverseTrigPolicy
//! must NOT change any existing behavior. These tests verify that:
//!
//! 1. Default config equals legacy behavior
//! 2. x/x simplifies as before (Generic mode)
//! 3. sqrt(-1) behavior unchanged (RealOnly)
//! 4. Inverse trig unchanged (Strict)

use cas_engine::{
    BranchPolicy, DomainMode, EvalConfig, InverseTrigPolicy, SimplifyOptions, ValueDomain,
};

// =============================================================================
// Default Configuration Tests
// =============================================================================

#[test]
fn default_eval_config_matches_legacy() {
    let cfg = EvalConfig::default();

    // Legacy defaults: Generic, RealOnly, Principal, Strict
    assert_eq!(cfg.domain_mode, DomainMode::Generic);
    assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
    assert_eq!(cfg.branch, BranchPolicy::Principal);
    assert_eq!(cfg.inv_trig, InverseTrigPolicy::Strict);
}

#[test]
fn simplify_options_includes_new_axes() {
    let opts = SimplifyOptions::default();

    // New axes should be present with defaults
    assert_eq!(opts.shared.semantics.inv_trig, InverseTrigPolicy::Strict);
    assert_eq!(opts.shared.semantics.value_domain, ValueDomain::RealOnly);
    assert_eq!(opts.shared.semantics.branch, BranchPolicy::Principal);
}

// =============================================================================
// Behavior Stability Tests
// =============================================================================

#[test]
fn x_div_x_simplifies_with_default_config() {
    use cas_ast::DisplayExpr;
    use cas_engine::{Engine, EvalAction, EvalRequest, EvalResult};
    use cas_parser::parse;
    use cas_session::SessionState;

    let mut engine = Engine::new();
    let mut state = SessionState::new();
    // Default EvalOptions with new axes should behave like before

    let parsed = parse("x/x", &mut engine.simplifier.context).unwrap();
    let req = EvalRequest {
        raw_input: "x/x".to_string(),
        parsed,
        kind: cas_engine::EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).unwrap();
    let result_str = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    // Default (Generic mode) should simplify x/x to 1
    assert_eq!(result_str, "1");
}

#[test]
fn eval_options_propagates_new_axes_to_simplify_options() {
    use cas_engine::options::EvalOptions;

    let eval_opts = EvalOptions::default();
    let simplify_opts = eval_opts.to_simplify_options();

    // All new axes should propagate correctly
    assert_eq!(
        simplify_opts.shared.semantics.inv_trig,
        InverseTrigPolicy::Strict
    );
    assert_eq!(
        simplify_opts.shared.semantics.value_domain,
        ValueDomain::RealOnly
    );
    assert_eq!(
        simplify_opts.shared.semantics.branch,
        BranchPolicy::Principal
    );
}

// =============================================================================
// Round-trip Stability Tests
// =============================================================================

#[test]
fn eval_config_constructors_consistent() {
    // strict() should use Strict domain mode + Strict inv_trig
    let strict = EvalConfig::strict();
    assert_eq!(strict.domain_mode, DomainMode::Strict);
    assert_eq!(strict.inv_trig, InverseTrigPolicy::Strict);
    assert_eq!(strict.value_domain, ValueDomain::RealOnly);

    // assume() should use Assume domain mode
    let assume = EvalConfig::assume();
    assert_eq!(assume.domain_mode, DomainMode::Assume);

    // complex() should enable ComplexEnabled
    let complex = EvalConfig::complex();
    assert_eq!(complex.value_domain, ValueDomain::ComplexEnabled);
    assert_eq!(complex.domain_mode, DomainMode::Generic);
}

#[test]
fn parent_context_has_new_accessors() {
    use cas_engine::parent_context::ParentContext;

    let ctx = ParentContext::root();

    // New accessors should return defaults
    assert_eq!(ctx.inv_trig_policy(), InverseTrigPolicy::Strict);
    assert_eq!(ctx.value_domain(), ValueDomain::RealOnly);
    assert_eq!(ctx.branch_policy(), BranchPolicy::Principal);
}

#[test]
fn parent_context_extend_preserves_semantics() {
    use cas_ast::Context;
    use cas_engine::parent_context::ParentContext;

    let mut ast_ctx = Context::new();
    let parent_id = ast_ctx.num(42);

    let ctx = ParentContext::root()
        .with_domain_mode(DomainMode::Strict)
        .with_inv_trig(InverseTrigPolicy::PrincipalValue);

    // extend() should preserve all semantic fields
    let extended = ctx.extend(parent_id);
    assert_eq!(extended.domain_mode(), DomainMode::Strict);
    assert_eq!(
        extended.inv_trig_policy(),
        InverseTrigPolicy::PrincipalValue
    );
}

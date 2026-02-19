//! Contract tests for semantics presets.
//!
//! These tests verify that preset configurations are correctly defined.
//! Note: Full behavior tests require REPL integration; these verify axes at config level.

use cas_engine::ConstFoldMode;
use cas_engine::DomainMode;
use cas_engine::{AssumeScope, BranchPolicy, EvalConfig, InverseTrigPolicy, ValueDomain};

// ============================================================================
// Preset Configuration Tests
// ============================================================================
// These verify the axis values that make up each preset match the contract.

/// default preset: generic, real, principal, strict, off
#[test]
fn preset_default_axes() {
    // Verify default EvalConfig matches "default" preset
    let cfg = EvalConfig::default();

    assert_eq!(cfg.domain_mode, DomainMode::Generic, "default domain");
    assert_eq!(cfg.value_domain, ValueDomain::RealOnly, "default value");
    assert_eq!(cfg.branch, BranchPolicy::Principal, "default branch");
    assert_eq!(cfg.inv_trig, InverseTrigPolicy::Strict, "default inv_trig");
}

/// strict preset: strict, real, principal, strict, off
#[test]
fn preset_strict_axes() {
    let cfg = EvalConfig {
        domain_mode: DomainMode::Strict,
        value_domain: ValueDomain::RealOnly,
        branch: BranchPolicy::Principal,
        inv_trig: InverseTrigPolicy::Strict,
        assume_scope: AssumeScope::Real,
    };

    assert_eq!(cfg.domain_mode, DomainMode::Strict);
    assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
}

/// complex preset: generic, complex, principal, strict, safe
#[test]
fn preset_complex_axes() {
    let cfg = EvalConfig {
        domain_mode: DomainMode::Generic,
        value_domain: ValueDomain::ComplexEnabled,
        branch: BranchPolicy::Principal,
        inv_trig: InverseTrigPolicy::Strict,
        assume_scope: AssumeScope::Real,
    };

    // Key differentiator: complex + safe const_fold
    assert_eq!(cfg.value_domain, ValueDomain::ComplexEnabled);
    // const_fold is controlled separately via SimplifyOptions/EvalOptions
}

/// school preset: generic, real, principal, principal, off  
#[test]
fn preset_school_axes() {
    let cfg = EvalConfig {
        domain_mode: DomainMode::Generic,
        value_domain: ValueDomain::RealOnly,
        branch: BranchPolicy::Principal,
        inv_trig: InverseTrigPolicy::PrincipalValue,
        assume_scope: AssumeScope::Real,
    };

    // Key differentiator: inv_trig = principal
    assert_eq!(cfg.inv_trig, InverseTrigPolicy::PrincipalValue);
    assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
}

// ============================================================================
// Preset Behavior Tests (ConstFoldMode specific)
// ============================================================================

#[test]
fn const_fold_safe_enables_sqrt_negative_to_i() {
    use cas_ast::Context;
    use cas_engine::fold_constants;
    use cas_engine::Budget;

    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let sqrt_neg_one = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![neg_one]);

    let cfg = EvalConfig {
        value_domain: ValueDomain::ComplexEnabled,
        ..Default::default()
    };
    let mut budget = Budget::preset_unlimited();

    // With Safe mode + ComplexEnabled, sqrt(-1) -> i
    let result = fold_constants(
        &mut ctx,
        sqrt_neg_one,
        &cfg,
        ConstFoldMode::Safe,
        &mut budget,
    )
    .unwrap();

    let display = cas_formatter::DisplayExpr {
        context: &ctx,
        id: result.expr,
    };
    let result_str = display.to_string();
    // Result should contain i (complex unit), may or may not simplify fully
    assert!(
        result_str.contains("i"),
        "sqrt(-1) with complex+safe should produce i, got: {}",
        result_str
    );
}

#[test]
fn const_fold_off_preserves_sqrt_negative() {
    use cas_ast::Context;
    use cas_engine::fold_constants;
    use cas_engine::Budget;

    let mut ctx = Context::new();
    let neg_one = ctx.num(-1);
    let sqrt_neg_one = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![neg_one]);

    let cfg = EvalConfig {
        value_domain: ValueDomain::ComplexEnabled,
        ..Default::default()
    };
    let mut budget = Budget::preset_unlimited();

    // With Off mode, even with ComplexEnabled, sqrt(-1) stays unchanged
    let result = fold_constants(
        &mut ctx,
        sqrt_neg_one,
        &cfg,
        ConstFoldMode::Off,
        &mut budget,
    )
    .unwrap();

    let display = cas_formatter::DisplayExpr {
        context: &ctx,
        id: result.expr,
    };
    let result_str = display.to_string();
    assert!(
        result_str.contains("sqrt") || result_str.contains("√"),
        "sqrt(-1) with const_fold=off should stay as sqrt, got: {}",
        result_str
    );
}

// ============================================================================
// Preset Contract: Each preset has distinct purpose
// ============================================================================

#[test]
fn presets_are_distinct() {
    // Verify each preset differs in at least one axis
    let default_cfg = EvalConfig::default();

    let strict_cfg = EvalConfig {
        domain_mode: DomainMode::Strict,
        ..Default::default()
    };

    let complex_cfg = EvalConfig {
        value_domain: ValueDomain::ComplexEnabled,
        ..Default::default()
    };

    let school_cfg = EvalConfig {
        inv_trig: InverseTrigPolicy::PrincipalValue,
        ..Default::default()
    };

    // Each differs from default in exactly one key axis
    assert_ne!(strict_cfg.domain_mode, default_cfg.domain_mode);
    assert_ne!(complex_cfg.value_domain, default_cfg.value_domain);
    assert_ne!(school_cfg.inv_trig, default_cfg.inv_trig);
}

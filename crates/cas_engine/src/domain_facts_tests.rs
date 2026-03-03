use crate::assumptions::ConditionClass;
use crate::domain::{DomainMode, Proof};
use crate::domain_facts::{
    decide, mode_allows_predicate, proof_to_strength, strength_to_proof, FactStrength, Predicate,
};

#[test]
fn test_proof_to_strength_mapping() {
    assert_eq!(proof_to_strength(Proof::Proven), FactStrength::Proven);
    assert_eq!(
        proof_to_strength(Proof::ProvenImplicit),
        FactStrength::Proven
    );
    assert_eq!(proof_to_strength(Proof::Unknown), FactStrength::Unknown);
    assert_eq!(proof_to_strength(Proof::Disproven), FactStrength::Disproven);
}

#[test]
fn test_strength_to_proof_roundtrip() {
    // Proven -> Proof::Proven (not ProvenImplicit)
    assert_eq!(strength_to_proof(FactStrength::Proven), Proof::Proven);
    assert_eq!(strength_to_proof(FactStrength::Unknown), Proof::Unknown);
    assert_eq!(strength_to_proof(FactStrength::Disproven), Proof::Disproven);
}

#[test]
fn test_predicate_condition_class() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");

    assert_eq!(
        Predicate::NonZero(x).condition_class(),
        ConditionClass::Definability
    );
    assert_eq!(
        Predicate::Defined(x).condition_class(),
        ConditionClass::Definability
    );
    assert_eq!(
        Predicate::Positive(x).condition_class(),
        ConditionClass::Analytic
    );
    assert_eq!(
        Predicate::NonNegative(x).condition_class(),
        ConditionClass::Analytic
    );
}

#[test]
fn test_mode_allows_predicate() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");

    let nonzero = Predicate::NonZero(x);
    let positive = Predicate::Positive(x);

    // Strict: blocks everything
    assert!(!mode_allows_predicate(DomainMode::Strict, &nonzero));
    assert!(!mode_allows_predicate(DomainMode::Strict, &positive));

    // Generic: allows Definability, blocks Analytic
    assert!(mode_allows_predicate(DomainMode::Generic, &nonzero));
    assert!(!mode_allows_predicate(DomainMode::Generic, &positive));

    // Assume: allows everything
    assert!(mode_allows_predicate(DomainMode::Assume, &nonzero));
    assert!(mode_allows_predicate(DomainMode::Assume, &positive));
}

#[test]
fn test_decide_proven() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let pred = Predicate::NonZero(x);

    // Proven facts are always allowed, regardless of mode
    let decision = decide(DomainMode::Strict, &pred, FactStrength::Proven);
    assert!(decision.allow);
}

#[test]
fn test_decide_disproven() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let pred = Predicate::NonZero(x);

    // Disproven facts are never allowed, regardless of mode
    let decision = decide(DomainMode::Assume, &pred, FactStrength::Disproven);
    assert!(!decision.allow);
}

#[test]
fn test_decide_unknown_nonzero() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let pred = Predicate::NonZero(x);

    // NonZero is Definability: blocked in Strict, allowed in Generic/Assume
    assert!(!decide(DomainMode::Strict, &pred, FactStrength::Unknown).allow);
    assert!(decide(DomainMode::Generic, &pred, FactStrength::Unknown).allow);
    assert!(decide(DomainMode::Assume, &pred, FactStrength::Unknown).allow);
}

#[test]
fn test_decide_unknown_positive() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let pred = Predicate::Positive(x);

    // Positive is Analytic: blocked in Strict+Generic, allowed only in Assume
    assert!(!decide(DomainMode::Strict, &pred, FactStrength::Unknown).allow);
    assert!(!decide(DomainMode::Generic, &pred, FactStrength::Unknown).allow);
    assert!(decide(DomainMode::Assume, &pred, FactStrength::Unknown).allow);
}

#[test]
fn test_decide_matches_can_cancel_factor() {
    // Verify that decide() produces the same result as can_cancel_factor()
    // for the NonZero predicate (Definability class)
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let pred = Predicate::NonZero(x);

    for mode in [DomainMode::Strict, DomainMode::Generic, DomainMode::Assume] {
        for proof in [
            Proof::Proven,
            Proof::ProvenImplicit,
            Proof::Unknown,
            Proof::Disproven,
        ] {
            let strength = proof_to_strength(proof);
            let new_decision = decide(mode, &pred, strength);
            let old_decision = crate::domain::can_cancel_factor(mode, proof);
            assert_eq!(
                new_decision.allow, old_decision.allow,
                "Mismatch for mode={:?}, proof={:?}: new={}, old={}",
                mode, proof, new_decision.allow, old_decision.allow,
            );
        }
    }
}

#[test]
fn test_fact_strength_helpers() {
    assert!(FactStrength::Proven.is_proven());
    assert!(!FactStrength::Proven.is_unknown());
    assert!(!FactStrength::Proven.is_disproven());

    assert!(!FactStrength::Unknown.is_proven());
    assert!(FactStrength::Unknown.is_unknown());

    assert!(!FactStrength::Disproven.is_proven());
    assert!(FactStrength::Disproven.is_disproven());
}

#[test]
fn test_predicate_describe() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");

    assert_eq!(Predicate::NonZero(x).describe(), "≠ 0");
    assert_eq!(Predicate::Positive(x).describe(), "> 0");
    assert_eq!(Predicate::NonNegative(x).describe(), "≥ 0");
    assert_eq!(Predicate::Defined(x).describe(), "is defined");
}

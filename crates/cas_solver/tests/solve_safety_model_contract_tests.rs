use cas_solver::{ConditionClass, DomainMode, Provenance, SolveSafety};

#[test]
fn test_always_safe_everywhere() {
    let safety = SolveSafety::Always;
    assert!(safety.safe_for_prepass());
    assert!(safety.safe_for_tactic(DomainMode::Strict));
    assert!(safety.safe_for_tactic(DomainMode::Generic));
    assert!(safety.safe_for_tactic(DomainMode::Assume));
}

#[test]
fn test_intrinsic_condition_allowed_in_generic() {
    let safety = SolveSafety::IntrinsicCondition(ConditionClass::Analytic);
    assert!(
        !safety.safe_for_prepass(),
        "Intrinsic still blocked in prepass"
    );
    assert!(
        !safety.safe_for_tactic(DomainMode::Strict),
        "Strict requires proof"
    );
    assert!(
        safety.safe_for_tactic(DomainMode::Generic),
        "Generic inherits intrinsic"
    );
    assert!(
        safety.safe_for_tactic(DomainMode::Assume),
        "Assume allows all"
    );
}

#[test]
fn test_definability_blocked_in_prepass() {
    let safety = SolveSafety::NeedsCondition(ConditionClass::Definability);
    assert!(!safety.safe_for_prepass());
    // Definability allowed in Generic and Assume.
    assert!(!safety.safe_for_tactic(DomainMode::Strict));
    assert!(safety.safe_for_tactic(DomainMode::Generic));
    assert!(safety.safe_for_tactic(DomainMode::Assume));
}

#[test]
fn test_analytic_blocked_in_prepass() {
    let safety = SolveSafety::NeedsCondition(ConditionClass::Analytic);
    assert!(!safety.safe_for_prepass());
    // Analytic only allowed in Assume (introduced, not intrinsic).
    assert!(!safety.safe_for_tactic(DomainMode::Strict));
    assert!(!safety.safe_for_tactic(DomainMode::Generic));
    assert!(safety.safe_for_tactic(DomainMode::Assume));
}

#[test]
fn test_never_blocked_everywhere() {
    let safety = SolveSafety::Never;
    assert!(!safety.safe_for_prepass());
    assert!(!safety.safe_for_tactic(DomainMode::Strict));
    assert!(!safety.safe_for_tactic(DomainMode::Generic));
    assert!(!safety.safe_for_tactic(DomainMode::Assume));
}

#[test]
fn test_always_has_no_descriptor() {
    assert!(SolveSafety::Always.requirement_descriptor().is_none());
}

#[test]
fn test_never_has_no_descriptor() {
    assert!(SolveSafety::Never.requirement_descriptor().is_none());
}

#[test]
fn test_intrinsic_analytic_descriptor() {
    let desc = SolveSafety::IntrinsicCondition(ConditionClass::Analytic)
        .requirement_descriptor()
        .expect("IntrinsicCondition should produce a descriptor");
    assert_eq!(desc.class, ConditionClass::Analytic);
    assert_eq!(desc.provenance, Provenance::Intrinsic);
}

#[test]
fn test_intrinsic_definability_descriptor() {
    let desc = SolveSafety::IntrinsicCondition(ConditionClass::Definability)
        .requirement_descriptor()
        .expect("IntrinsicCondition should produce a descriptor");
    assert_eq!(desc.class, ConditionClass::Definability);
    assert_eq!(desc.provenance, Provenance::Intrinsic);
}

#[test]
fn test_needs_analytic_descriptor() {
    let desc = SolveSafety::NeedsCondition(ConditionClass::Analytic)
        .requirement_descriptor()
        .expect("NeedsCondition should produce a descriptor");
    assert_eq!(desc.class, ConditionClass::Analytic);
    assert_eq!(desc.provenance, Provenance::Introduced);
}

#[test]
fn test_needs_definability_descriptor() {
    let desc = SolveSafety::NeedsCondition(ConditionClass::Definability)
        .requirement_descriptor()
        .expect("NeedsCondition should produce a descriptor");
    assert_eq!(desc.class, ConditionClass::Definability);
    assert_eq!(desc.provenance, Provenance::Introduced);
}

#[test]
fn test_descriptor_distinguishes_intrinsic_from_introduced() {
    let intrinsic = SolveSafety::IntrinsicCondition(ConditionClass::Analytic)
        .requirement_descriptor()
        .unwrap();
    let introduced = SolveSafety::NeedsCondition(ConditionClass::Analytic)
        .requirement_descriptor()
        .unwrap();
    // Same class, different provenance.
    assert_eq!(intrinsic.class, introduced.class);
    assert_ne!(intrinsic.provenance, introduced.provenance);
}

#[test]
fn test_solve_safety_roundtrip_with_engine_type() {
    let local = SolveSafety::NeedsCondition(ConditionClass::Definability);
    let engine: cas_engine::SolveSafety = local.into_engine();
    let mapped_back = SolveSafety::from(engine);
    assert_eq!(local, mapped_back);
}

#[test]
fn test_requirement_descriptor_roundtrip_with_engine_type() {
    let local = SolveSafety::IntrinsicCondition(ConditionClass::Analytic)
        .requirement_descriptor()
        .unwrap();
    let engine: cas_engine::RequirementDescriptor = local.into();
    let mapped_back: cas_solver::RequirementDescriptor = engine.into();
    assert_eq!(local, mapped_back);
}

#[test]
fn test_assumption_record_roundtrip_with_engine_type() {
    let local = cas_solver::AssumptionRecord {
        kind: "nonzero".to_string(),
        expr: "x".to_string(),
        message: "x != 0".to_string(),
        count: 2,
    };
    let engine: cas_engine::AssumptionRecord = local.clone().into();
    let mapped_back: cas_solver::AssumptionRecord = engine.into();
    assert_eq!(local, mapped_back);
}

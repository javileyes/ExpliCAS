use crate::domain::DomainMode;
use crate::semantics::{BranchPolicy, EvalConfig, InverseTrigPolicy, ValueDomain};

#[test]
fn test_default_config() {
    let cfg = EvalConfig::default();
    assert_eq!(cfg.domain_mode, DomainMode::Generic);
    assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
    assert_eq!(cfg.branch, BranchPolicy::Principal);
    assert_eq!(cfg.inv_trig, InverseTrigPolicy::Strict);
}

#[test]
fn test_strict_config() {
    let cfg = EvalConfig::strict();
    assert_eq!(cfg.domain_mode, DomainMode::Strict);
    assert_eq!(cfg.inv_trig, InverseTrigPolicy::Strict);
}

#[test]
fn test_assume_config() {
    let cfg = EvalConfig::assume();
    assert_eq!(cfg.domain_mode, DomainMode::Assume);
}

#[test]
fn test_complex_config() {
    let cfg = EvalConfig::complex();
    assert_eq!(cfg.value_domain, ValueDomain::ComplexEnabled);
}

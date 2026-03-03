use crate::domain::DomainMode;

#[test]
fn test_domain_mode_default_is_generic() {
    assert_eq!(DomainMode::default(), DomainMode::Generic);
}

#[test]
fn test_domain_mode_predicates() {
    assert!(DomainMode::Strict.is_strict());
    assert!(!DomainMode::Strict.is_generic());

    assert!(DomainMode::Generic.is_generic());
    assert!(!DomainMode::Generic.is_strict());

    assert!(DomainMode::Assume.is_assume());
    assert!(!DomainMode::Assume.is_strict());
}

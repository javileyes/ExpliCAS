/// Parse domain mode from JSON string axis.
pub(crate) fn domain_mode_from_str(value: &str) -> crate::DomainMode {
    match value {
        "strict" => crate::DomainMode::Strict,
        "assume" => crate::DomainMode::Assume,
        _ => crate::DomainMode::Generic,
    }
}

/// Parse value domain from JSON string axis.
pub(crate) fn value_domain_from_str(value: &str) -> crate::ValueDomain {
    match value {
        "complex" => crate::ValueDomain::ComplexEnabled,
        _ => crate::ValueDomain::RealOnly,
    }
}

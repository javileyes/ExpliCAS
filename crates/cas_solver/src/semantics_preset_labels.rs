pub(crate) fn domain_mode_label(value: crate::DomainMode) -> &'static str {
    match value {
        crate::DomainMode::Strict => "strict",
        crate::DomainMode::Generic => "generic",
        crate::DomainMode::Assume => "assume",
    }
}

pub(crate) fn value_domain_label(value: crate::ValueDomain) -> &'static str {
    match value {
        crate::ValueDomain::RealOnly => "real",
        crate::ValueDomain::ComplexEnabled => "complex",
    }
}

pub(crate) fn branch_policy_label(value: crate::BranchPolicy) -> &'static str {
    match value {
        crate::BranchPolicy::Principal => "principal",
    }
}

pub(crate) fn inverse_trig_policy_label(value: crate::InverseTrigPolicy) -> &'static str {
    match value {
        crate::InverseTrigPolicy::Strict => "strict",
        crate::InverseTrigPolicy::PrincipalValue => "principal",
    }
}

pub(crate) fn const_fold_mode_label(value: crate::ConstFoldMode) -> &'static str {
    match value {
        crate::ConstFoldMode::Off => "off",
        crate::ConstFoldMode::Safe => "safe",
    }
}

pub(crate) fn domain_mode_label(value: cas_solver::DomainMode) -> &'static str {
    match value {
        cas_solver::DomainMode::Strict => "strict",
        cas_solver::DomainMode::Generic => "generic",
        cas_solver::DomainMode::Assume => "assume",
    }
}

pub(crate) fn value_domain_label(value: cas_solver::ValueDomain) -> &'static str {
    match value {
        cas_solver::ValueDomain::RealOnly => "real",
        cas_solver::ValueDomain::ComplexEnabled => "complex",
    }
}

pub(crate) fn branch_policy_label(value: cas_solver::BranchPolicy) -> &'static str {
    match value {
        cas_solver::BranchPolicy::Principal => "principal",
    }
}

pub(crate) fn inverse_trig_policy_label(value: cas_solver::InverseTrigPolicy) -> &'static str {
    match value {
        cas_solver::InverseTrigPolicy::Strict => "strict",
        cas_solver::InverseTrigPolicy::PrincipalValue => "principal",
    }
}

pub(crate) fn const_fold_mode_label(value: cas_solver::ConstFoldMode) -> &'static str {
    match value {
        cas_solver::ConstFoldMode::Off => "off",
        cas_solver::ConstFoldMode::Safe => "safe",
    }
}

use crate::semantics_view_types::SemanticsViewState;

pub(super) fn format_evaluation_overview_lines(state: &SemanticsViewState) -> Vec<String> {
    let domain = match state.domain_mode {
        crate::DomainMode::Strict => "strict",
        crate::DomainMode::Assume => "assume",
        crate::DomainMode::Generic => "generic",
    };

    let value = match state.value_domain {
        crate::ValueDomain::RealOnly => "real",
        crate::ValueDomain::ComplexEnabled => "complex",
    };

    let branch = match state.branch {
        crate::BranchPolicy::Principal => "principal",
    };

    let inv_trig = match state.inv_trig {
        crate::InverseTrigPolicy::Strict => "strict",
        crate::InverseTrigPolicy::PrincipalValue => "principal",
    };

    let const_fold = match state.const_fold {
        crate::ConstFoldMode::Off => "off",
        crate::ConstFoldMode::Safe => "safe",
    };

    let mut lines = vec![
        format!("  domain_mode: {}", domain),
        format!("  value_domain: {}", value),
    ];

    if state.value_domain == crate::ValueDomain::RealOnly {
        lines.push(format!(
            "  branch: {} (inactive: value_domain=real)",
            branch
        ));
    } else {
        lines.push(format!("  branch: {}", branch));
    }

    lines.push(format!("  inv_trig: {}", inv_trig));
    lines.push(format!("  const_fold: {}", const_fold));
    lines
}

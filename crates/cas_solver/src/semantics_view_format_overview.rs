use crate::semantics_view_types::SemanticsViewState;

/// Format all semantic axis settings.
pub fn format_semantics_overview_lines(state: &SemanticsViewState) -> Vec<String> {
    let mut lines = Vec::new();

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

    lines.push("Semantics:".to_string());
    lines.push(format!("  domain_mode: {}", domain));
    lines.push(format!("  value_domain: {}", value));

    if state.value_domain == crate::ValueDomain::RealOnly {
        lines.push(format!(
            "  branch: {} (inactive: value_domain=real)",
            branch
        ));
    } else {
        lines.push(format!("  branch: {}", branch));
    }

    lines.push(format!("  inv_trig: {}", inv_trig));

    let const_fold = match state.const_fold {
        crate::ConstFoldMode::Off => "off",
        crate::ConstFoldMode::Safe => "safe",
    };
    lines.push(format!("  const_fold: {}", const_fold));

    let assumptions = match state.assumption_reporting {
        crate::AssumptionReporting::Off => "off",
        crate::AssumptionReporting::Summary => "summary",
        crate::AssumptionReporting::Trace => "trace",
    };
    lines.push(format!("  assumptions: {}", assumptions));

    let assume_scope = match state.assume_scope {
        crate::AssumeScope::Real => "real",
        crate::AssumeScope::Wildcard => "wildcard",
    };
    if state.domain_mode != crate::DomainMode::Assume {
        lines.push(format!(
            "  assume_scope: {} (inactive: domain_mode != assume)",
            assume_scope
        ));
    } else {
        lines.push(format!("  assume_scope: {}", assume_scope));
    }

    lines.push(format!(
        "  hints: {}",
        if state.hints_enabled { "on" } else { "off" }
    ));

    let requires = match state.requires_display {
        crate::RequiresDisplayLevel::Essential => "essential",
        crate::RequiresDisplayLevel::All => "all",
    };
    lines.push(format!("  requires: {}", requires));

    lines
}

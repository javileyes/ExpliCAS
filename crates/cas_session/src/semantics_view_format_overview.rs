use crate::semantics_view_types::SemanticsViewState;

/// Format all semantic axis settings.
pub fn format_semantics_overview_lines(state: &SemanticsViewState) -> Vec<String> {
    let mut lines = Vec::new();

    let domain = match state.domain_mode {
        cas_solver::DomainMode::Strict => "strict",
        cas_solver::DomainMode::Assume => "assume",
        cas_solver::DomainMode::Generic => "generic",
    };

    let value = match state.value_domain {
        cas_solver::ValueDomain::RealOnly => "real",
        cas_solver::ValueDomain::ComplexEnabled => "complex",
    };

    let branch = match state.branch {
        cas_solver::BranchPolicy::Principal => "principal",
    };

    let inv_trig = match state.inv_trig {
        cas_solver::InverseTrigPolicy::Strict => "strict",
        cas_solver::InverseTrigPolicy::PrincipalValue => "principal",
    };

    lines.push("Semantics:".to_string());
    lines.push(format!("  domain_mode: {}", domain));
    lines.push(format!("  value_domain: {}", value));

    if state.value_domain == cas_solver::ValueDomain::RealOnly {
        lines.push(format!(
            "  branch: {} (inactive: value_domain=real)",
            branch
        ));
    } else {
        lines.push(format!("  branch: {}", branch));
    }

    lines.push(format!("  inv_trig: {}", inv_trig));

    let const_fold = match state.const_fold {
        cas_solver::ConstFoldMode::Off => "off",
        cas_solver::ConstFoldMode::Safe => "safe",
    };
    lines.push(format!("  const_fold: {}", const_fold));

    let assumptions = match state.assumption_reporting {
        cas_solver::AssumptionReporting::Off => "off",
        cas_solver::AssumptionReporting::Summary => "summary",
        cas_solver::AssumptionReporting::Trace => "trace",
    };
    lines.push(format!("  assumptions: {}", assumptions));

    let assume_scope = match state.assume_scope {
        cas_solver::AssumeScope::Real => "real",
        cas_solver::AssumeScope::Wildcard => "wildcard",
    };
    if state.domain_mode != cas_solver::DomainMode::Assume {
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
        cas_solver::RequiresDisplayLevel::Essential => "essential",
        cas_solver::RequiresDisplayLevel::All => "all",
    };
    lines.push(format!("  requires: {}", requires));

    lines
}

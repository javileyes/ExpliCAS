use crate::semantics_view_types::SemanticsViewState;

pub(super) fn format_reporting_overview_lines(state: &SemanticsViewState) -> Vec<String> {
    let assumptions = match state.assumption_reporting {
        crate::AssumptionReporting::Off => "off",
        crate::AssumptionReporting::Summary => "summary",
        crate::AssumptionReporting::Trace => "trace",
    };

    let assume_scope = match state.assume_scope {
        crate::AssumeScope::Real => "real",
        crate::AssumeScope::Wildcard => "wildcard",
    };

    let requires = match state.requires_display {
        crate::RequiresDisplayLevel::Essential => "essential",
        crate::RequiresDisplayLevel::All => "all",
    };

    let mut lines = vec![format!("  assumptions: {}", assumptions)];

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
    lines.push(format!("  requires: {}", requires));
    lines
}

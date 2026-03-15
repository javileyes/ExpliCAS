use crate::SemanticsViewState;

pub(super) fn format_reporting_axis_lines(state: &SemanticsViewState, axis: &str) -> Vec<String> {
    let mut lines = Vec::new();

    match axis {
        "assumptions" => {
            let current = match state.assumption_reporting {
                crate::AssumptionReporting::Off => "off",
                crate::AssumptionReporting::Summary => "summary",
                crate::AssumptionReporting::Trace => "trace",
            };
            lines.push(format!("assumptions: {}", current));
            lines.push("  Values: off | summary | trace".to_string());
            lines.push("  off:     No assumption reporting".to_string());
            lines.push("  summary: Deduped summary line at end".to_string());
            lines.push("  trace:   Detailed trace (future)".to_string());
        }
        "assume_scope" => {
            let current = match state.assume_scope {
                crate::AssumeScope::Real => "real",
                crate::AssumeScope::Wildcard => "wildcard",
            };
            let inactive = state.domain_mode != crate::DomainMode::Assume;
            if inactive {
                lines.push(format!(
                    "assume_scope: {} (inactive: domain_mode != assume)",
                    current
                ));
            } else {
                lines.push(format!("assume_scope: {}", current));
            }
            lines.push("  Values: real | wildcard".to_string());
            lines.push("  real:     Assume for ℝ, error if ℂ needed".to_string());
            lines.push("  wildcard: Assume for ℝ, residual+warning if ℂ needed".to_string());
            if inactive {
                lines.push("  Note: Only active when domain_mode=assume".to_string());
            }
        }
        "requires" => {
            let current = match state.requires_display {
                crate::RequiresDisplayLevel::Essential => "essential",
                crate::RequiresDisplayLevel::All => "all",
            };
            lines.push(format!("requires: {}", current));
            lines.push("  Values: essential | all".to_string());
            lines.push("  essential: Only show requires whose witness was consumed".to_string());
            lines.push("  all:       Show all requires including implicit ones".to_string());
        }
        _ => unreachable!("unsupported reporting axis: {axis}"),
    }

    lines
}

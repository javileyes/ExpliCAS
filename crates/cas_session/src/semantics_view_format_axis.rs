use crate::semantics_view_types::SemanticsViewState;

/// Format one semantic axis description and values.
pub fn format_semantics_axis_lines(state: &SemanticsViewState, axis: &str) -> Vec<String> {
    let mut lines = Vec::new();

    match axis {
        "domain" => {
            let current = match state.domain_mode {
                cas_solver::DomainMode::Strict => "strict",
                cas_solver::DomainMode::Assume => "assume",
                cas_solver::DomainMode::Generic => "generic",
            };
            lines.push(format!("domain: {}", current));
            lines.push("  Values: strict | generic | assume".to_string());
            lines.push("  strict:  No domain assumptions (x/x stays x/x)".to_string());
            lines.push("  generic: Classic CAS 'almost everywhere' algebra".to_string());
            lines.push("  assume:  Use assumptions with warnings".to_string());
        }
        "value" => {
            let current = match state.value_domain {
                cas_solver::ValueDomain::RealOnly => "real",
                cas_solver::ValueDomain::ComplexEnabled => "complex",
            };
            lines.push(format!("value: {}", current));
            lines.push("  Values: real | complex".to_string());
            lines.push("  real:    ℝ only (sqrt(-1) undefined)".to_string());
            lines.push("  complex: ℂ enabled (sqrt(-1) = i)".to_string());
        }
        "branch" => {
            let current = match state.branch {
                cas_solver::BranchPolicy::Principal => "principal",
            };
            let inactive = state.value_domain == cas_solver::ValueDomain::RealOnly;
            if inactive {
                lines.push(format!("branch: {} (inactive: value=real)", current));
            } else {
                lines.push(format!("branch: {}", current));
            }
            lines.push("  Values: principal".to_string());
            lines.push("  principal: Use principal branch for multi-valued functions".to_string());
            if inactive {
                lines.push("  Note: Only active when value=complex".to_string());
            }
        }
        "inv_trig" => {
            let current = match state.inv_trig {
                cas_solver::InverseTrigPolicy::Strict => "strict",
                cas_solver::InverseTrigPolicy::PrincipalValue => "principal",
            };
            lines.push(format!("inv_trig: {}", current));
            lines.push("  Values: strict | principal".to_string());
            lines.push("  strict:    arctan(tan(x)) unchanged".to_string());
            lines.push("  principal: arctan(tan(x)) → x with warning".to_string());
        }
        "const_fold" => {
            let current = match state.const_fold {
                cas_solver::ConstFoldMode::Off => "off",
                cas_solver::ConstFoldMode::Safe => "safe",
            };
            lines.push(format!("const_fold: {}", current));
            lines.push("  Values: off | safe".to_string());
            lines.push("  off:  No constant folding (defer semantic decisions)".to_string());
            lines.push("  safe: Fold literals (2^3 → 8, sqrt(-1) → i if complex)".to_string());
        }
        "assumptions" => {
            let current = match state.assumption_reporting {
                cas_solver::AssumptionReporting::Off => "off",
                cas_solver::AssumptionReporting::Summary => "summary",
                cas_solver::AssumptionReporting::Trace => "trace",
            };
            lines.push(format!("assumptions: {}", current));
            lines.push("  Values: off | summary | trace".to_string());
            lines.push("  off:     No assumption reporting".to_string());
            lines.push("  summary: Deduped summary line at end".to_string());
            lines.push("  trace:   Detailed trace (future)".to_string());
        }
        "assume_scope" => {
            let current = match state.assume_scope {
                cas_solver::AssumeScope::Real => "real",
                cas_solver::AssumeScope::Wildcard => "wildcard",
            };
            let inactive = state.domain_mode != cas_solver::DomainMode::Assume;
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
                cas_solver::RequiresDisplayLevel::Essential => "essential",
                cas_solver::RequiresDisplayLevel::All => "all",
            };
            lines.push(format!("requires: {}", current));
            lines.push("  Values: essential | all".to_string());
            lines.push("  essential: Only show requires whose witness was consumed".to_string());
            lines.push("  all:       Show all requires including implicit ones".to_string());
        }
        _ => {
            lines.push(format!("Unknown axis: {}", axis));
        }
    }

    lines
}

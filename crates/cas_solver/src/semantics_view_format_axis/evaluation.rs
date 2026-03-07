use crate::semantics_view_types::SemanticsViewState;

pub(super) fn format_evaluation_axis_lines(state: &SemanticsViewState, axis: &str) -> Vec<String> {
    let mut lines = Vec::new();

    match axis {
        "domain" => {
            let current = match state.domain_mode {
                crate::DomainMode::Strict => "strict",
                crate::DomainMode::Assume => "assume",
                crate::DomainMode::Generic => "generic",
            };
            lines.push(format!("domain: {}", current));
            lines.push("  Values: strict | generic | assume".to_string());
            lines.push("  strict:  No domain assumptions (x/x stays x/x)".to_string());
            lines.push("  generic: Classic CAS 'almost everywhere' algebra".to_string());
            lines.push("  assume:  Use assumptions with warnings".to_string());
        }
        "value" => {
            let current = match state.value_domain {
                crate::ValueDomain::RealOnly => "real",
                crate::ValueDomain::ComplexEnabled => "complex",
            };
            lines.push(format!("value: {}", current));
            lines.push("  Values: real | complex".to_string());
            lines.push("  real:    ℝ only (sqrt(-1) undefined)".to_string());
            lines.push("  complex: ℂ enabled (sqrt(-1) = i)".to_string());
        }
        "branch" => {
            let current = match state.branch {
                crate::BranchPolicy::Principal => "principal",
            };
            let inactive = state.value_domain == crate::ValueDomain::RealOnly;
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
                crate::InverseTrigPolicy::Strict => "strict",
                crate::InverseTrigPolicy::PrincipalValue => "principal",
            };
            lines.push(format!("inv_trig: {}", current));
            lines.push("  Values: strict | principal".to_string());
            lines.push("  strict:    arctan(tan(x)) unchanged".to_string());
            lines.push("  principal: arctan(tan(x)) → x with warning".to_string());
        }
        "const_fold" => {
            let current = match state.const_fold {
                crate::ConstFoldMode::Off => "off",
                crate::ConstFoldMode::Safe => "safe",
            };
            lines.push(format!("const_fold: {}", current));
            lines.push("  Values: off | safe".to_string());
            lines.push("  off:  No constant folding (defer semantic decisions)".to_string());
            lines.push("  safe: Fold literals (2^3 → 8, sqrt(-1) → i if complex)".to_string());
        }
        _ => unreachable!("unsupported evaluation axis: {axis}"),
    }

    lines
}

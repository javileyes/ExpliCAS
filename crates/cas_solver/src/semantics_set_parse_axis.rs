use crate::semantics_set_types::SemanticsSetState;

pub(crate) fn set_semantic_axis(
    state: &mut SemanticsSetState,
    axis: &str,
    value: &str,
) -> Option<String> {
    match axis {
        "domain" => match value {
            "strict" => state.domain_mode = crate::DomainMode::Strict,
            "generic" => state.domain_mode = crate::DomainMode::Generic,
            "assume" => state.domain_mode = crate::DomainMode::Assume,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'domain'\nAllowed: strict, generic, assume",
                    value
                ));
            }
        },
        "value" => match value {
            "real" => state.value_domain = crate::ValueDomain::RealOnly,
            "complex" => state.value_domain = crate::ValueDomain::ComplexEnabled,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'value'\nAllowed: real, complex",
                    value
                ));
            }
        },
        "branch" => match value {
            "principal" => state.branch = crate::BranchPolicy::Principal,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'branch'\nAllowed: principal",
                    value
                ));
            }
        },
        "inv_trig" => match value {
            "strict" => state.inv_trig = crate::InverseTrigPolicy::Strict,
            "principal" => state.inv_trig = crate::InverseTrigPolicy::PrincipalValue,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'inv_trig'\nAllowed: strict, principal",
                    value
                ));
            }
        },
        "const_fold" => match value {
            "off" => state.const_fold = crate::ConstFoldMode::Off,
            "safe" => state.const_fold = crate::ConstFoldMode::Safe,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'const_fold'\nAllowed: off, safe",
                    value
                ));
            }
        },
        "assumptions" => match value {
            "off" => state.assumption_reporting = crate::AssumptionReporting::Off,
            "summary" => state.assumption_reporting = crate::AssumptionReporting::Summary,
            "trace" => state.assumption_reporting = crate::AssumptionReporting::Trace,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'assumptions'\nAllowed: off, summary, trace",
                    value
                ));
            }
        },
        "assume_scope" => match value {
            "real" => state.assume_scope = crate::AssumeScope::Real,
            "wildcard" => state.assume_scope = crate::AssumeScope::Wildcard,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'assume_scope'\nAllowed: real, wildcard",
                    value
                ));
            }
        },
        "hints" => match value {
            "on" => state.hints_enabled = true,
            "off" => state.hints_enabled = false,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'hints'\nAllowed: on, off",
                    value
                ));
            }
        },
        "solve" => match value {
            "check" => {
                return Some(
                    "ERROR: Use 'semantics set solve check on' or 'semantics set solve check off'"
                        .to_string(),
                );
            }
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'solve'\nAllowed: 'check on', 'check off'",
                    value
                ));
            }
        },
        "requires" => match value {
            "essential" => state.requires_display = crate::RequiresDisplayLevel::Essential,
            "all" => state.requires_display = crate::RequiresDisplayLevel::All,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'requires'\nAllowed: essential, all",
                    value
                ));
            }
        },
        _ => {
            return Some(format!(
                "ERROR: Unknown axis '{}'\n\
                 Valid axes: domain, value, branch, inv_trig, const_fold, assumptions, assume_scope, hints, solve, requires",
                axis
            ));
        }
    }
    None
}

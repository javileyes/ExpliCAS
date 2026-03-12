use cas_solver_core::semantics_set_types::SemanticsSetState;

pub(super) fn set_evaluation_axis(
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
        _ => unreachable!("unsupported evaluation axis: {axis}"),
    }

    None
}

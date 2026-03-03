use crate::semantics_set_types::SemanticsSetState;

fn set_semantic_axis(state: &mut SemanticsSetState, axis: &str, value: &str) -> Option<String> {
    match axis {
        "domain" => match value {
            "strict" => state.domain_mode = cas_solver::DomainMode::Strict,
            "generic" => state.domain_mode = cas_solver::DomainMode::Generic,
            "assume" => state.domain_mode = cas_solver::DomainMode::Assume,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'domain'\nAllowed: strict, generic, assume",
                    value
                ));
            }
        },
        "value" => match value {
            "real" => state.value_domain = cas_solver::ValueDomain::RealOnly,
            "complex" => state.value_domain = cas_solver::ValueDomain::ComplexEnabled,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'value'\nAllowed: real, complex",
                    value
                ));
            }
        },
        "branch" => match value {
            "principal" => state.branch = cas_solver::BranchPolicy::Principal,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'branch'\nAllowed: principal",
                    value
                ));
            }
        },
        "inv_trig" => match value {
            "strict" => state.inv_trig = cas_solver::InverseTrigPolicy::Strict,
            "principal" => state.inv_trig = cas_solver::InverseTrigPolicy::PrincipalValue,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'inv_trig'\nAllowed: strict, principal",
                    value
                ));
            }
        },
        "const_fold" => match value {
            "off" => state.const_fold = cas_solver::ConstFoldMode::Off,
            "safe" => state.const_fold = cas_solver::ConstFoldMode::Safe,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'const_fold'\nAllowed: off, safe",
                    value
                ));
            }
        },
        "assumptions" => match value {
            "off" => state.assumption_reporting = cas_solver::AssumptionReporting::Off,
            "summary" => state.assumption_reporting = cas_solver::AssumptionReporting::Summary,
            "trace" => state.assumption_reporting = cas_solver::AssumptionReporting::Trace,
            _ => {
                return Some(format!(
                    "ERROR: Invalid value '{}' for axis 'assumptions'\nAllowed: off, summary, trace",
                    value
                ));
            }
        },
        "assume_scope" => match value {
            "real" => state.assume_scope = cas_solver::AssumeScope::Real,
            "wildcard" => state.assume_scope = cas_solver::AssumeScope::Wildcard,
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
            "essential" => state.requires_display = cas_solver::RequiresDisplayLevel::Essential,
            "all" => state.requires_display = cas_solver::RequiresDisplayLevel::All,
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

/// Parse and apply arguments from `semantics set ...`.
pub fn evaluate_semantics_set_args(
    args: &[&str],
    mut state: SemanticsSetState,
) -> Result<SemanticsSetState, String> {
    if args.is_empty() {
        return Err("Usage: semantics set <axis> <value>\n\
                   or:  semantics set <axis>=<value> ..."
            .to_string());
    }

    let mut i = 0;
    while i < args.len() {
        let arg = args[i];

        if let Some((key, value)) = arg.split_once('=') {
            if let Some(err) = set_semantic_axis(&mut state, key, value) {
                return Err(err);
            }
            i += 1;
            continue;
        }

        if i + 1 >= args.len() {
            return Err(format!("ERROR: Missing value for axis '{}'", arg));
        }

        if arg == "solve" && args.get(i + 1) == Some(&"check") && i + 2 < args.len() {
            match args[i + 2] {
                "on" => state.check_solutions = true,
                "off" => state.check_solutions = false,
                other => {
                    return Err(format!(
                        "ERROR: Invalid value '{}' for 'solve check'\nAllowed: on, off",
                        other
                    ));
                }
            }
            i += 3;
            continue;
        }

        if let Some(err) = set_semantic_axis(&mut state, arg, args[i + 1]) {
            return Err(err);
        }
        i += 2;
    }

    Ok(state)
}

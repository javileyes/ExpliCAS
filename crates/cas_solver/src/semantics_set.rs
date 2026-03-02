/// Mutable semantics state for evaluating `semantics set` commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsSetState {
    pub domain_mode: crate::DomainMode,
    pub value_domain: crate::ValueDomain,
    pub branch: crate::BranchPolicy,
    pub inv_trig: crate::InverseTrigPolicy,
    pub const_fold: crate::ConstFoldMode,
    pub assumption_reporting: crate::AssumptionReporting,
    pub assume_scope: crate::AssumeScope,
    pub hints_enabled: bool,
    pub check_solutions: bool,
    pub requires_display: crate::RequiresDisplayLevel,
}

/// Build a mutable semantics-set snapshot from simplifier + eval options.
pub fn semantics_set_state_from_options(
    simplify_options: &crate::SimplifyOptions,
    eval_options: &crate::EvalOptions,
) -> SemanticsSetState {
    SemanticsSetState {
        domain_mode: simplify_options.shared.semantics.domain_mode,
        value_domain: simplify_options.shared.semantics.value_domain,
        branch: simplify_options.shared.semantics.branch,
        inv_trig: simplify_options.shared.semantics.inv_trig,
        const_fold: eval_options.const_fold,
        assumption_reporting: eval_options.shared.assumption_reporting,
        assume_scope: simplify_options.shared.semantics.assume_scope,
        hints_enabled: eval_options.hints_enabled,
        check_solutions: eval_options.check_solutions,
        requires_display: eval_options.requires_display,
    }
}

/// Apply semantic state to both simplifier options and runtime eval options.
pub fn apply_semantics_set_state_to_options(
    next: SemanticsSetState,
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) {
    simplify_options.shared.semantics.domain_mode = next.domain_mode;
    simplify_options.shared.semantics.value_domain = next.value_domain;
    simplify_options.shared.semantics.branch = next.branch;
    simplify_options.shared.semantics.inv_trig = next.inv_trig;
    simplify_options.shared.semantics.assume_scope = next.assume_scope;
    simplify_options.shared.assumption_reporting = next.assumption_reporting;

    eval_options.shared.semantics.domain_mode = next.domain_mode;
    eval_options.shared.semantics.value_domain = next.value_domain;
    eval_options.shared.semantics.branch = next.branch;
    eval_options.shared.semantics.inv_trig = next.inv_trig;
    eval_options.shared.semantics.assume_scope = next.assume_scope;
    eval_options.shared.assumption_reporting = next.assumption_reporting;

    eval_options.const_fold = next.const_fold;
    eval_options.hints_enabled = next.hints_enabled;
    eval_options.check_solutions = next.check_solutions;
    eval_options.requires_display = next.requires_display;
}

fn set_semantic_axis(state: &mut SemanticsSetState, axis: &str, value: &str) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::{
        apply_semantics_set_state_to_options, evaluate_semantics_set_args,
        semantics_set_state_from_options, SemanticsSetState,
    };

    fn state() -> SemanticsSetState {
        SemanticsSetState {
            domain_mode: crate::DomainMode::Generic,
            value_domain: crate::ValueDomain::RealOnly,
            branch: crate::BranchPolicy::Principal,
            inv_trig: crate::InverseTrigPolicy::Strict,
            const_fold: crate::ConstFoldMode::Off,
            assumption_reporting: crate::AssumptionReporting::Summary,
            assume_scope: crate::AssumeScope::Real,
            hints_enabled: true,
            check_solutions: true,
            requires_display: crate::RequiresDisplayLevel::Essential,
        }
    }

    #[test]
    fn evaluate_semantics_set_args_updates_key_value_pairs() {
        let next = evaluate_semantics_set_args(&["domain=strict", "value=complex"], state())
            .expect("should parse");
        assert_eq!(next.domain_mode, crate::DomainMode::Strict);
        assert_eq!(next.value_domain, crate::ValueDomain::ComplexEnabled);
    }

    #[test]
    fn evaluate_semantics_set_args_supports_solve_check_triplet() {
        let next =
            evaluate_semantics_set_args(&["solve", "check", "off"], state()).expect("should parse");
        assert!(!next.check_solutions);
    }

    #[test]
    fn evaluate_semantics_set_args_rejects_invalid_axis() {
        let err = evaluate_semantics_set_args(&["nope", "x"], state()).expect_err("should fail");
        assert!(err.contains("ERROR: Unknown axis"));
    }

    #[test]
    fn apply_semantics_set_state_to_options_updates_shared_semantics() {
        let mut simplify_options = crate::SimplifyOptions::default();
        let mut eval_options = crate::EvalOptions::default();
        let next = SemanticsSetState {
            domain_mode: crate::DomainMode::Strict,
            value_domain: crate::ValueDomain::ComplexEnabled,
            branch: crate::BranchPolicy::Principal,
            inv_trig: crate::InverseTrigPolicy::PrincipalValue,
            const_fold: crate::ConstFoldMode::Safe,
            assumption_reporting: crate::AssumptionReporting::Trace,
            assume_scope: crate::AssumeScope::Wildcard,
            hints_enabled: false,
            check_solutions: false,
            requires_display: crate::RequiresDisplayLevel::All,
        };
        apply_semantics_set_state_to_options(next, &mut simplify_options, &mut eval_options);

        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            crate::DomainMode::Strict
        );
        assert_eq!(
            eval_options.shared.semantics.value_domain,
            crate::ValueDomain::ComplexEnabled
        );
        assert_eq!(eval_options.const_fold, crate::ConstFoldMode::Safe);
        assert!(!eval_options.hints_enabled);
        assert_eq!(
            eval_options.requires_display,
            crate::RequiresDisplayLevel::All
        );
    }

    #[test]
    fn semantics_set_state_from_options_reads_check_solutions() {
        let simplify_options = crate::SimplifyOptions::default();
        let eval_options = crate::EvalOptions {
            check_solutions: false,
            ..crate::EvalOptions::default()
        };
        let state = semantics_set_state_from_options(&simplify_options, &eval_options);
        assert!(!state.check_solutions);
    }
}

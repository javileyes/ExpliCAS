#[cfg(test)]
mod tests {
    use crate::{
        apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
        evaluate_semantics_set_args, evaluate_semantics_set_args_to_overview_lines,
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

    #[test]
    fn apply_semantics_set_args_to_options_updates_runtime_state() {
        let mut simplify_options = crate::SimplifyOptions::default();
        let mut eval_options = crate::EvalOptions::default();

        let next = apply_semantics_set_args_to_options(
            &["domain", "assume", "assumptions", "trace"],
            &mut simplify_options,
            &mut eval_options,
        )
        .expect("should parse and apply");

        assert_eq!(next.domain_mode, crate::DomainMode::Assume);
        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            crate::DomainMode::Assume
        );
        assert_eq!(
            eval_options.shared.assumption_reporting,
            crate::AssumptionReporting::Trace
        );
    }

    #[test]
    fn evaluate_semantics_set_args_to_overview_lines_returns_overview() {
        let mut simplify_options = crate::SimplifyOptions::default();
        let mut eval_options = crate::EvalOptions::default();
        let lines = evaluate_semantics_set_args_to_overview_lines(
            &["domain", "assume"],
            &mut simplify_options,
            &mut eval_options,
        )
        .expect("should parse and format");
        assert!(lines.iter().any(|line| line.contains("domain_mode")));
        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            crate::DomainMode::Assume
        );
    }
}

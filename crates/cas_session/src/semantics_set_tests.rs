#[cfg(test)]
mod tests {
    use crate::{
        apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
        evaluate_semantics_set_args, evaluate_semantics_set_args_to_overview_lines,
        semantics_set_state_from_options, SemanticsSetState,
    };

    fn state() -> SemanticsSetState {
        SemanticsSetState {
            domain_mode: cas_solver::DomainMode::Generic,
            value_domain: cas_solver::ValueDomain::RealOnly,
            branch: cas_solver::BranchPolicy::Principal,
            inv_trig: cas_solver::InverseTrigPolicy::Strict,
            const_fold: cas_solver::ConstFoldMode::Off,
            assumption_reporting: cas_solver::AssumptionReporting::Summary,
            assume_scope: cas_solver::AssumeScope::Real,
            hints_enabled: true,
            check_solutions: true,
            requires_display: cas_solver::RequiresDisplayLevel::Essential,
        }
    }

    #[test]
    fn evaluate_semantics_set_args_updates_key_value_pairs() {
        let next = evaluate_semantics_set_args(&["domain=strict", "value=complex"], state())
            .expect("should parse");
        assert_eq!(next.domain_mode, cas_solver::DomainMode::Strict);
        assert_eq!(next.value_domain, cas_solver::ValueDomain::ComplexEnabled);
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
        let mut simplify_options = cas_solver::SimplifyOptions::default();
        let mut eval_options = cas_solver::EvalOptions::default();
        let next = SemanticsSetState {
            domain_mode: cas_solver::DomainMode::Strict,
            value_domain: cas_solver::ValueDomain::ComplexEnabled,
            branch: cas_solver::BranchPolicy::Principal,
            inv_trig: cas_solver::InverseTrigPolicy::PrincipalValue,
            const_fold: cas_solver::ConstFoldMode::Safe,
            assumption_reporting: cas_solver::AssumptionReporting::Trace,
            assume_scope: cas_solver::AssumeScope::Wildcard,
            hints_enabled: false,
            check_solutions: false,
            requires_display: cas_solver::RequiresDisplayLevel::All,
        };
        apply_semantics_set_state_to_options(next, &mut simplify_options, &mut eval_options);

        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            cas_solver::DomainMode::Strict
        );
        assert_eq!(
            eval_options.shared.semantics.value_domain,
            cas_solver::ValueDomain::ComplexEnabled
        );
        assert_eq!(eval_options.const_fold, cas_solver::ConstFoldMode::Safe);
        assert!(!eval_options.hints_enabled);
        assert_eq!(
            eval_options.requires_display,
            cas_solver::RequiresDisplayLevel::All
        );
    }

    #[test]
    fn semantics_set_state_from_options_reads_check_solutions() {
        let simplify_options = cas_solver::SimplifyOptions::default();
        let eval_options = cas_solver::EvalOptions {
            check_solutions: false,
            ..cas_solver::EvalOptions::default()
        };
        let state = semantics_set_state_from_options(&simplify_options, &eval_options);
        assert!(!state.check_solutions);
    }

    #[test]
    fn apply_semantics_set_args_to_options_updates_runtime_state() {
        let mut simplify_options = cas_solver::SimplifyOptions::default();
        let mut eval_options = cas_solver::EvalOptions::default();

        let next = apply_semantics_set_args_to_options(
            &["domain", "assume", "assumptions", "trace"],
            &mut simplify_options,
            &mut eval_options,
        )
        .expect("should parse and apply");

        assert_eq!(next.domain_mode, cas_solver::DomainMode::Assume);
        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            cas_solver::DomainMode::Assume
        );
        assert_eq!(
            eval_options.shared.assumption_reporting,
            cas_solver::AssumptionReporting::Trace
        );
    }

    #[test]
    fn evaluate_semantics_set_args_to_overview_lines_returns_overview() {
        let mut simplify_options = cas_solver::SimplifyOptions::default();
        let mut eval_options = cas_solver::EvalOptions::default();
        let lines = evaluate_semantics_set_args_to_overview_lines(
            &["domain", "assume"],
            &mut simplify_options,
            &mut eval_options,
        )
        .expect("should parse and format");
        assert!(lines.iter().any(|line| line.contains("domain_mode")));
        assert_eq!(
            simplify_options.shared.semantics.domain_mode,
            cas_solver::DomainMode::Assume
        );
    }
}

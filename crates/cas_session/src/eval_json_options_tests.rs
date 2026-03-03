#[cfg(test)]
mod tests {
    use crate::eval_json_options::{apply_eval_json_options, EvalJsonOptionAxes};

    #[test]
    fn apply_eval_json_options_maps_known_axes() {
        let mut opts = cas_solver::EvalOptions::default();
        apply_eval_json_options(
            &mut opts,
            EvalJsonOptionAxes {
                context: "solve",
                branch: "principal",
                complex: "on",
                autoexpand: "auto",
                steps: "compact",
                domain: "strict",
                value_domain: "complex",
                inv_trig: "principal",
                complex_branch: "principal",
                assume_scope: "wildcard",
            },
        );

        assert_eq!(opts.shared.context_mode, cas_solver::ContextMode::Solve);
        assert_eq!(opts.branch_mode, cas_solver::BranchMode::PrincipalBranch);
        assert_eq!(opts.complex_mode, cas_solver::ComplexMode::On);
        assert_eq!(opts.shared.expand_policy, cas_solver::ExpandPolicy::Auto);
        assert_eq!(opts.steps_mode, cas_solver::StepsMode::Compact);
        assert_eq!(
            opts.shared.semantics.domain_mode,
            cas_solver::DomainMode::Strict
        );
        assert_eq!(
            opts.shared.semantics.value_domain,
            cas_solver::ValueDomain::ComplexEnabled
        );
        assert_eq!(
            opts.shared.semantics.inv_trig,
            cas_solver::InverseTrigPolicy::PrincipalValue
        );
        assert_eq!(
            opts.shared.semantics.assume_scope,
            cas_solver::AssumeScope::Wildcard
        );
    }
}

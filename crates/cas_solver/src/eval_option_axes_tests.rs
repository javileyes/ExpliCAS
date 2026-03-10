#[cfg(test)]
mod tests {
    use crate::eval_option_axes::{apply_eval_option_axes, EvalOptionAxes};

    #[test]
    fn apply_eval_option_axes_maps_known_axes() {
        let mut opts = crate::EvalOptions::default();
        apply_eval_option_axes(
            &mut opts,
            EvalOptionAxes {
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

        assert_eq!(opts.shared.context_mode, crate::ContextMode::Solve);
        assert_eq!(opts.branch_mode, crate::BranchMode::PrincipalBranch);
        assert_eq!(opts.complex_mode, crate::ComplexMode::On);
        assert_eq!(opts.shared.expand_policy, crate::ExpandPolicy::Auto);
        assert_eq!(opts.steps_mode, crate::StepsMode::Compact);
        assert_eq!(opts.shared.semantics.domain_mode, crate::DomainMode::Strict);
        assert_eq!(
            opts.shared.semantics.value_domain,
            crate::ValueDomain::ComplexEnabled
        );
        assert_eq!(
            opts.shared.semantics.inv_trig,
            crate::InverseTrigPolicy::PrincipalValue
        );
        assert_eq!(
            opts.shared.semantics.assume_scope,
            crate::AssumeScope::Wildcard
        );
    }
}

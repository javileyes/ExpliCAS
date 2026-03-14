#[cfg(test)]
mod tests {
    use crate::eval_option_axes::{apply_eval_option_axes, EvalOptionAxes};

    #[test]
    fn apply_eval_option_axes_maps_known_axes() {
        let mut opts = crate::EvalOptions::default();
        apply_eval_option_axes(
            &mut opts,
            EvalOptionAxes {
                context: cas_api_models::EvalContextMode::Solve,
                branch: cas_api_models::EvalBranchMode::Principal,
                complex: cas_api_models::EvalComplexMode::On,
                const_fold: cas_api_models::EvalConstFoldMode::Safe,
                autoexpand: cas_api_models::EvalExpandPolicy::Auto,
                steps: cas_api_models::EvalStepsMode::Compact,
                domain: cas_api_models::EvalDomainMode::Strict,
                value_domain: cas_api_models::EvalValueDomain::Complex,
                inv_trig: cas_api_models::EvalInvTrigPolicy::Principal,
                complex_branch: cas_api_models::EvalBranchMode::Principal,
                assume_scope: cas_api_models::EvalAssumeScope::Wildcard,
            },
        );

        assert_eq!(opts.shared.context_mode, crate::ContextMode::Solve);
        assert_eq!(opts.branch_mode, crate::BranchMode::PrincipalBranch);
        assert_eq!(opts.complex_mode, crate::ComplexMode::On);
        assert_eq!(opts.const_fold, crate::ConstFoldMode::Safe);
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

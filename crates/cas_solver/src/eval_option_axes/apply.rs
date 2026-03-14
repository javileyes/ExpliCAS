use super::EvalOptionAxes;

fn axes_match_current_options(opts: &crate::EvalOptions, axes: EvalOptionAxes) -> bool {
    matches!(
        (opts.shared.context_mode, axes.context),
        (
            crate::ContextMode::Standard,
            cas_api_models::EvalContextMode::Standard
        ) | (
            crate::ContextMode::Solve,
            cas_api_models::EvalContextMode::Solve
        ) | (
            crate::ContextMode::IntegratePrep,
            cas_api_models::EvalContextMode::Integrate
        ) | (
            crate::ContextMode::Auto,
            cas_api_models::EvalContextMode::Auto
        )
    ) && matches!(
        (opts.branch_mode, axes.branch),
        (
            crate::BranchMode::PrincipalBranch,
            cas_api_models::EvalBranchMode::Principal
        ) | (
            crate::BranchMode::Strict,
            cas_api_models::EvalBranchMode::Strict
        )
    ) && matches!(
        (opts.complex_mode, axes.complex),
        (crate::ComplexMode::On, cas_api_models::EvalComplexMode::On)
            | (
                crate::ComplexMode::Off,
                cas_api_models::EvalComplexMode::Off
            )
            | (
                crate::ComplexMode::Auto,
                cas_api_models::EvalComplexMode::Auto
            )
    ) && matches!(
        (opts.const_fold, axes.const_fold),
        (
            crate::ConstFoldMode::Off,
            cas_api_models::EvalConstFoldMode::Off
        ) | (
            crate::ConstFoldMode::Safe,
            cas_api_models::EvalConstFoldMode::Safe
        )
    ) && matches!(
        (opts.steps_mode, axes.steps),
        (crate::StepsMode::On, cas_api_models::EvalStepsMode::On)
            | (
                crate::StepsMode::Compact,
                cas_api_models::EvalStepsMode::Compact
            )
            | (crate::StepsMode::Off, cas_api_models::EvalStepsMode::Off)
    ) && matches!(
        (opts.shared.expand_policy, axes.autoexpand),
        (
            crate::ExpandPolicy::Auto,
            cas_api_models::EvalExpandPolicy::Auto
        ) | (
            crate::ExpandPolicy::Off,
            cas_api_models::EvalExpandPolicy::Off
        )
    ) && matches!(
        (opts.shared.semantics.domain_mode, axes.domain),
        (
            crate::DomainMode::Strict,
            cas_api_models::EvalDomainMode::Strict
        ) | (
            crate::DomainMode::Generic,
            cas_api_models::EvalDomainMode::Generic
        ) | (
            crate::DomainMode::Assume,
            cas_api_models::EvalDomainMode::Assume
        )
    ) && matches!(
        (opts.shared.semantics.inv_trig, axes.inv_trig),
        (
            crate::InverseTrigPolicy::PrincipalValue,
            cas_api_models::EvalInvTrigPolicy::Principal
        ) | (
            crate::InverseTrigPolicy::Strict,
            cas_api_models::EvalInvTrigPolicy::Strict
        )
    ) && matches!(
        (opts.shared.semantics.value_domain, axes.value_domain),
        (
            crate::ValueDomain::ComplexEnabled,
            cas_api_models::EvalValueDomain::Complex
        ) | (
            crate::ValueDomain::RealOnly,
            cas_api_models::EvalValueDomain::Real
        )
    ) && matches!(
        (opts.shared.semantics.assume_scope, axes.assume_scope),
        (
            crate::AssumeScope::Wildcard,
            cas_api_models::EvalAssumeScope::Wildcard
        ) | (
            crate::AssumeScope::Real,
            cas_api_models::EvalAssumeScope::Real
        )
    )
}

/// Apply typed eval option axes onto `EvalOptions`.
pub(crate) fn apply_eval_option_axes(opts: &mut crate::EvalOptions, axes: EvalOptionAxes) {
    if axes_match_current_options(opts, axes) {
        return;
    }

    opts.shared.context_mode = match axes.context {
        cas_api_models::EvalContextMode::Standard => crate::ContextMode::Standard,
        cas_api_models::EvalContextMode::Solve => crate::ContextMode::Solve,
        cas_api_models::EvalContextMode::Integrate => crate::ContextMode::IntegratePrep,
        cas_api_models::EvalContextMode::Auto => crate::ContextMode::Auto,
    };

    opts.branch_mode = match axes.branch {
        cas_api_models::EvalBranchMode::Principal => crate::BranchMode::PrincipalBranch,
        cas_api_models::EvalBranchMode::Strict => crate::BranchMode::Strict,
    };

    opts.complex_mode = match axes.complex {
        cas_api_models::EvalComplexMode::On => crate::ComplexMode::On,
        cas_api_models::EvalComplexMode::Off => crate::ComplexMode::Off,
        cas_api_models::EvalComplexMode::Auto => crate::ComplexMode::Auto,
    };

    opts.const_fold = match axes.const_fold {
        cas_api_models::EvalConstFoldMode::Off => crate::ConstFoldMode::Off,
        cas_api_models::EvalConstFoldMode::Safe => crate::ConstFoldMode::Safe,
    };

    opts.steps_mode = match axes.steps {
        cas_api_models::EvalStepsMode::On => crate::StepsMode::On,
        cas_api_models::EvalStepsMode::Compact => crate::StepsMode::Compact,
        cas_api_models::EvalStepsMode::Off => crate::StepsMode::Off,
    };

    opts.shared.expand_policy = match axes.autoexpand {
        cas_api_models::EvalExpandPolicy::Auto => crate::ExpandPolicy::Auto,
        cas_api_models::EvalExpandPolicy::Off => crate::ExpandPolicy::Off,
    };

    opts.shared.semantics.domain_mode = match axes.domain {
        cas_api_models::EvalDomainMode::Strict => crate::DomainMode::Strict,
        cas_api_models::EvalDomainMode::Generic => crate::DomainMode::Generic,
        cas_api_models::EvalDomainMode::Assume => crate::DomainMode::Assume,
    };

    opts.shared.semantics.inv_trig = match axes.inv_trig {
        cas_api_models::EvalInvTrigPolicy::Principal => crate::InverseTrigPolicy::PrincipalValue,
        cas_api_models::EvalInvTrigPolicy::Strict => crate::InverseTrigPolicy::Strict,
    };

    opts.shared.semantics.value_domain = match axes.value_domain {
        cas_api_models::EvalValueDomain::Complex => crate::ValueDomain::ComplexEnabled,
        cas_api_models::EvalValueDomain::Real => crate::ValueDomain::RealOnly,
    };

    let _ = axes.complex_branch;
    opts.shared.semantics.branch = crate::BranchPolicy::Principal;

    opts.shared.semantics.assume_scope = match axes.assume_scope {
        cas_api_models::EvalAssumeScope::Wildcard => crate::AssumeScope::Wildcard,
        cas_api_models::EvalAssumeScope::Real => crate::AssumeScope::Real,
    };
}

use super::parse::{domain_mode_from_str, value_domain_from_str};
use super::types::EvalOptionAxes;

/// Apply stringly eval option axes onto `EvalOptions`.
pub(crate) fn apply_eval_option_axes(opts: &mut crate::EvalOptions, axes: EvalOptionAxes<'_>) {
    opts.shared.context_mode = match axes.context {
        "standard" => crate::ContextMode::Standard,
        "solve" => crate::ContextMode::Solve,
        "integrate" => crate::ContextMode::IntegratePrep,
        _ => crate::ContextMode::Auto,
    };

    opts.branch_mode = match axes.branch {
        "principal" => crate::BranchMode::PrincipalBranch,
        _ => crate::BranchMode::Strict,
    };

    opts.complex_mode = match axes.complex {
        "on" => crate::ComplexMode::On,
        "off" => crate::ComplexMode::Off,
        _ => crate::ComplexMode::Auto,
    };

    opts.steps_mode = match axes.steps {
        "on" => crate::StepsMode::On,
        "compact" => crate::StepsMode::Compact,
        _ => crate::StepsMode::Off,
    };

    opts.shared.expand_policy = match axes.autoexpand {
        "auto" => crate::ExpandPolicy::Auto,
        _ => crate::ExpandPolicy::Off,
    };

    opts.shared.semantics.domain_mode = domain_mode_from_str(axes.domain);

    opts.shared.semantics.inv_trig = match axes.inv_trig {
        "principal" => crate::InverseTrigPolicy::PrincipalValue,
        _ => crate::InverseTrigPolicy::Strict,
    };

    opts.shared.semantics.value_domain = value_domain_from_str(axes.value_domain);

    let _ = axes.complex_branch;
    opts.shared.semantics.branch = crate::BranchPolicy::Principal;

    opts.shared.semantics.assume_scope = match axes.assume_scope {
        "wildcard" => crate::AssumeScope::Wildcard,
        _ => crate::AssumeScope::Real,
    };
}

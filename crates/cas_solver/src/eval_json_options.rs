//! Eval-json option mapping helpers.

/// Stringly option axes accepted by eval-json command.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EvalJsonOptionAxes<'a> {
    pub context: &'a str,
    pub branch: &'a str,
    pub complex: &'a str,
    pub autoexpand: &'a str,
    pub steps: &'a str,
    pub domain: &'a str,
    pub value_domain: &'a str,
    pub inv_trig: &'a str,
    pub complex_branch: &'a str,
    pub assume_scope: &'a str,
}

/// Apply eval-json option axes onto `EvalOptions`.
pub(crate) fn apply_eval_json_options(opts: &mut crate::EvalOptions, axes: EvalJsonOptionAxes<'_>) {
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

    opts.shared.semantics.domain_mode = match axes.domain {
        "strict" => crate::DomainMode::Strict,
        "assume" => crate::DomainMode::Assume,
        _ => crate::DomainMode::Generic,
    };

    opts.shared.semantics.inv_trig = match axes.inv_trig {
        "principal" => crate::InverseTrigPolicy::PrincipalValue,
        _ => crate::InverseTrigPolicy::Strict,
    };

    opts.shared.semantics.value_domain = match axes.value_domain {
        "complex" => crate::ValueDomain::ComplexEnabled,
        _ => crate::ValueDomain::RealOnly,
    };

    let _ = axes.complex_branch;
    opts.shared.semantics.branch = crate::BranchPolicy::Principal;

    opts.shared.semantics.assume_scope = match axes.assume_scope {
        "wildcard" => crate::AssumeScope::Wildcard,
        _ => crate::AssumeScope::Real,
    };
}

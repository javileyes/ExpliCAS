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
pub(crate) fn apply_eval_json_options(
    opts: &mut cas_solver::EvalOptions,
    axes: EvalJsonOptionAxes<'_>,
) {
    opts.shared.context_mode = match axes.context {
        "standard" => cas_solver::ContextMode::Standard,
        "solve" => cas_solver::ContextMode::Solve,
        "integrate" => cas_solver::ContextMode::IntegratePrep,
        _ => cas_solver::ContextMode::Auto,
    };

    opts.branch_mode = match axes.branch {
        "principal" => cas_solver::BranchMode::PrincipalBranch,
        _ => cas_solver::BranchMode::Strict,
    };

    opts.complex_mode = match axes.complex {
        "on" => cas_solver::ComplexMode::On,
        "off" => cas_solver::ComplexMode::Off,
        _ => cas_solver::ComplexMode::Auto,
    };

    opts.steps_mode = match axes.steps {
        "on" => cas_solver::StepsMode::On,
        "compact" => cas_solver::StepsMode::Compact,
        _ => cas_solver::StepsMode::Off,
    };

    opts.shared.expand_policy = match axes.autoexpand {
        "auto" => cas_solver::ExpandPolicy::Auto,
        _ => cas_solver::ExpandPolicy::Off,
    };

    opts.shared.semantics.domain_mode = match axes.domain {
        "strict" => cas_solver::DomainMode::Strict,
        "assume" => cas_solver::DomainMode::Assume,
        _ => cas_solver::DomainMode::Generic,
    };

    opts.shared.semantics.inv_trig = match axes.inv_trig {
        "principal" => cas_solver::InverseTrigPolicy::PrincipalValue,
        _ => cas_solver::InverseTrigPolicy::Strict,
    };

    opts.shared.semantics.value_domain = match axes.value_domain {
        "complex" => cas_solver::ValueDomain::ComplexEnabled,
        _ => cas_solver::ValueDomain::RealOnly,
    };

    let _ = axes.complex_branch;
    opts.shared.semantics.branch = cas_solver::BranchPolicy::Principal;

    opts.shared.semantics.assume_scope = match axes.assume_scope {
        "wildcard" => cas_solver::AssumeScope::Wildcard,
        _ => cas_solver::AssumeScope::Real,
    };
}

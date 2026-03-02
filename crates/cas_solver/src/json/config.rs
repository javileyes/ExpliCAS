use crate::{
    AssumeScope, BranchMode, BranchPolicy, ComplexMode, ContextMode, ExpandPolicy, StepsMode,
};
use cas_api_models::{BudgetJsonInfo, DomainJson, OptionsJson, SemanticsJson};

/// Apply eval-json textual options to engine eval options.
///
/// This preserves CLI contract mapping for string flags.
#[allow(clippy::too_many_arguments)]
pub fn apply_eval_json_options(
    opts: &mut crate::EvalOptions,
    context: &str,
    branch: &str,
    complex: &str,
    autoexpand: &str,
    steps: &str,
    domain: &str,
    value_domain: &str,
    inv_trig: &str,
    complex_branch: &str,
    assume_scope: &str,
) {
    opts.shared.context_mode = match context {
        "standard" => ContextMode::Standard,
        "solve" => ContextMode::Solve,
        "integrate" => ContextMode::IntegratePrep,
        _ => ContextMode::Auto,
    };

    opts.branch_mode = match branch {
        "principal" => BranchMode::PrincipalBranch,
        _ => BranchMode::Strict,
    };

    opts.complex_mode = match complex {
        "on" => ComplexMode::On,
        "off" => ComplexMode::Off,
        _ => ComplexMode::Auto,
    };

    opts.steps_mode = match steps {
        "on" => StepsMode::On,
        "compact" => StepsMode::Compact,
        _ => StepsMode::Off,
    };

    opts.shared.expand_policy = match autoexpand {
        "auto" => ExpandPolicy::Auto,
        _ => ExpandPolicy::Off,
    };

    opts.shared.semantics.domain_mode = match domain {
        "strict" => crate::DomainMode::Strict,
        "assume" => crate::DomainMode::Assume,
        _ => crate::DomainMode::Generic,
    };

    opts.shared.semantics.inv_trig = match inv_trig {
        "principal" => crate::InverseTrigPolicy::PrincipalValue,
        _ => crate::InverseTrigPolicy::Strict,
    };

    opts.shared.semantics.value_domain = match value_domain {
        "complex" => crate::ValueDomain::ComplexEnabled,
        _ => crate::ValueDomain::RealOnly,
    };

    let _ = complex_branch;
    opts.shared.semantics.branch = BranchPolicy::Principal;

    opts.shared.semantics.assume_scope = match assume_scope {
        "wildcard" => AssumeScope::Wildcard,
        _ => AssumeScope::Real,
    };
}

pub fn build_options_json_eval(
    context: &str,
    branch: &str,
    autoexpand: &str,
    complex: &str,
    steps: &str,
    domain: &str,
    const_fold: &str,
) -> OptionsJson {
    OptionsJson {
        context_mode: context.to_string(),
        branch_mode: branch.to_string(),
        expand_policy: autoexpand.to_string(),
        complex_mode: complex.to_string(),
        steps_mode: steps.to_string(),
        domain_mode: domain.to_string(),
        const_fold: const_fold.to_string(),
    }
}

pub fn build_budget_json_eval(budget_preset: &str, strict: bool) -> BudgetJsonInfo {
    BudgetJsonInfo::new(budget_preset, strict)
}

pub fn build_domain_json_eval(domain: &str) -> DomainJson {
    DomainJson {
        mode: domain.to_string(),
    }
}

pub fn build_semantics_json_eval(
    domain: &str,
    value_domain: &str,
    complex_branch: &str,
    inv_trig: &str,
    assume_scope: &str,
) -> SemanticsJson {
    SemanticsJson {
        domain_mode: domain.to_string(),
        value_domain: value_domain.to_string(),
        branch: complex_branch.to_string(),
        inv_trig: inv_trig.to_string(),
        assume_scope: assume_scope.to_string(),
    }
}

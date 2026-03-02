#[test]
fn apply_eval_json_options_maps_known_values() {
    let mut opts = cas_solver::EvalOptions::default();
    cas_solver::json::apply_eval_json_options(
        &mut opts,
        "integrate",
        "principal",
        "on",
        "auto",
        "compact",
        "assume",
        "complex",
        "principal",
        "principal",
        "wildcard",
    );

    assert_eq!(
        opts.shared.context_mode,
        cas_solver::ContextMode::IntegratePrep
    );
    assert_eq!(opts.branch_mode, cas_solver::BranchMode::PrincipalBranch);
    assert_eq!(opts.complex_mode, cas_solver::ComplexMode::On);
    assert_eq!(opts.shared.expand_policy, cas_solver::ExpandPolicy::Auto);
    assert_eq!(opts.steps_mode, cas_solver::StepsMode::Compact);
    assert_eq!(
        opts.shared.semantics.domain_mode,
        cas_solver::DomainMode::Assume
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
        opts.shared.semantics.branch,
        cas_solver::BranchPolicy::Principal
    );
    assert_eq!(
        opts.shared.semantics.assume_scope,
        cas_solver::AssumeScope::Wildcard
    );
}

#[test]
fn apply_eval_json_options_falls_back_to_defaults_for_unknown_values() {
    let mut opts = cas_solver::EvalOptions::default();
    cas_solver::json::apply_eval_json_options(
        &mut opts, "??", "??", "??", "??", "??", "??", "??", "??", "??", "??",
    );

    assert_eq!(opts.shared.context_mode, cas_solver::ContextMode::Auto);
    assert_eq!(opts.branch_mode, cas_solver::BranchMode::Strict);
    assert_eq!(opts.complex_mode, cas_solver::ComplexMode::Auto);
    assert_eq!(opts.shared.expand_policy, cas_solver::ExpandPolicy::Off);
    assert_eq!(opts.steps_mode, cas_solver::StepsMode::Off);
    assert_eq!(
        opts.shared.semantics.domain_mode,
        cas_solver::DomainMode::Generic
    );
    assert_eq!(
        opts.shared.semantics.value_domain,
        cas_solver::ValueDomain::RealOnly
    );
    assert_eq!(
        opts.shared.semantics.inv_trig,
        cas_solver::InverseTrigPolicy::Strict
    );
    assert_eq!(
        opts.shared.semantics.branch,
        cas_solver::BranchPolicy::Principal
    );
    assert_eq!(
        opts.shared.semantics.assume_scope,
        cas_solver::AssumeScope::Real
    );
}

#[test]
fn build_eval_json_metadata_matches_input_strings() {
    let options = cas_solver::json::build_options_json_eval(
        "auto", "strict", "off", "auto", "on", "generic", "safe",
    );
    assert_eq!(options.context_mode, "auto");
    assert_eq!(options.branch_mode, "strict");
    assert_eq!(options.expand_policy, "off");
    assert_eq!(options.complex_mode, "auto");
    assert_eq!(options.steps_mode, "on");
    assert_eq!(options.domain_mode, "generic");
    assert_eq!(options.const_fold, "safe");

    let budget = cas_solver::json::build_budget_json_eval("standard", true);
    assert_eq!(budget.preset, "standard");
    assert_eq!(budget.mode, "strict");
    assert!(budget.exceeded.is_none());

    let domain = cas_solver::json::build_domain_json_eval("strict");
    assert_eq!(domain.mode, "strict");

    let semantics = cas_solver::json::build_semantics_json_eval(
        "strict",
        "complex",
        "principal",
        "strict",
        "real",
    );
    assert_eq!(semantics.domain_mode, "strict");
    assert_eq!(semantics.value_domain, "complex");
    assert_eq!(semantics.branch, "principal");
    assert_eq!(semantics.inv_trig, "strict");
    assert_eq!(semantics.assume_scope, "real");
}

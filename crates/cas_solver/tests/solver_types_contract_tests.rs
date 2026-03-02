use cas_ast::Context;
use cas_solver::{
    AssumeScope, AssumptionEvent, DomainMode, EvalOptions, ImplicitCondition, SolveCtx,
    SolveDomainEnv, SolverOptions, ValueDomain,
};
use cas_solver_core::log_domain::DomainModeKind;
use cas_solver_core::solve_budget::SolveBudget;
use cas_solver_core::strategy_options::pow_kernel_inputs;

#[test]
fn solve_ctx_fork_shares_required_conditions() {
    let mut context = Context::new();
    let x = context.var("x");

    let parent = SolveCtx::default();
    parent.note_required_condition(ImplicitCondition::NonZero(x));

    let child = parent.fork_with_domain_env_next_depth(SolveDomainEnv::default());
    child.note_required_condition(ImplicitCondition::Positive(x));

    let snapshot = parent.snapshot();
    assert_eq!(snapshot.required.len(), 2);
}

#[test]
fn solve_ctx_fork_shares_assumptions_and_scopes() {
    let mut context = Context::new();
    let x = context.var("x");

    let parent = SolveCtx::default();
    let child = parent.fork_with_domain_env_next_depth(SolveDomainEnv::default());
    child.note_assumption(AssumptionEvent::positive(&context, x));
    child.emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
        "QuadraticFormula",
    ));

    let snapshot = parent.snapshot();
    assert_eq!(snapshot.assumed.len(), 1);
    assert_eq!(snapshot.output_scopes.len(), 1);
}

#[test]
fn solve_ctx_fork_increments_depth() {
    let parent = SolveCtx::default();
    assert_eq!(parent.depth(), 0);

    let child = parent.fork_with_domain_env_next_depth(SolveDomainEnv::default());
    let grandchild = child.fork_with_domain_env_next_depth(SolveDomainEnv::default());

    assert_eq!(child.depth(), 1);
    assert_eq!(grandchild.depth(), 2);
}

#[test]
fn solver_option_branch_and_tactic_flags() {
    let opts = SolverOptions {
        domain_mode: DomainMode::Assume,
        value_domain: ValueDomain::RealOnly,
        budget: SolveBudget {
            max_branches: 3,
            ..Default::default()
        },
        ..Default::default()
    };

    let inputs = pow_kernel_inputs(
        opts.core_domain_mode(),
        opts.wildcard_scope(),
        opts.value_domain == ValueDomain::RealOnly,
        opts.budget,
    );
    assert!(inputs.shortcut_can_branch);
    assert!(inputs.log_can_branch);
    assert!(inputs.solve_tactic_enabled);
    assert_eq!(inputs.mode, DomainModeKind::Assume);
}

#[test]
fn solver_options_from_eval_options_maps_semantics_and_budget() {
    let mut eval_opts = EvalOptions::default();
    eval_opts.shared.semantics.value_domain = ValueDomain::ComplexEnabled;
    eval_opts.shared.semantics.domain_mode = DomainMode::Assume;
    eval_opts.shared.semantics.assume_scope = AssumeScope::Wildcard;
    eval_opts.budget = SolveBudget {
        max_branches: 5,
        max_depth: 4,
    };

    let opts = SolverOptions::from_eval_options(&eval_opts);
    assert_eq!(opts.value_domain, ValueDomain::ComplexEnabled);
    assert_eq!(opts.domain_mode, DomainMode::Assume);
    assert_eq!(opts.assume_scope, AssumeScope::Wildcard);
    assert_eq!(opts.budget.max_branches, 5);
    assert_eq!(opts.budget.max_depth, 4);
}

#[test]
fn solver_options_into_engine_preserves_fields() {
    let opts = SolverOptions {
        value_domain: ValueDomain::ComplexEnabled,
        domain_mode: DomainMode::Assume,
        assume_scope: AssumeScope::Wildcard,
        budget: SolveBudget {
            max_branches: 7,
            max_depth: 6,
        },
        detailed_steps: false,
    };

    let engine_opts = opts.into_engine();
    assert_eq!(engine_opts.value_domain, ValueDomain::ComplexEnabled);
    assert_eq!(engine_opts.domain_mode, DomainMode::Assume);
    assert_eq!(engine_opts.assume_scope, AssumeScope::Wildcard);
    assert_eq!(engine_opts.budget.max_branches, 7);
    assert_eq!(engine_opts.budget.max_depth, 6);
    assert!(!engine_opts.detailed_steps);
}

#[test]
fn solver_options_from_engine_preserves_fields() {
    let engine_opts = cas_engine::SolverOptions {
        value_domain: ValueDomain::ComplexEnabled,
        domain_mode: DomainMode::Strict,
        assume_scope: AssumeScope::Real,
        budget: SolveBudget {
            max_branches: 2,
            max_depth: 3,
        },
        detailed_steps: true,
    };

    let opts = SolverOptions::from(engine_opts);
    assert_eq!(opts.value_domain, ValueDomain::ComplexEnabled);
    assert_eq!(opts.domain_mode, DomainMode::Strict);
    assert_eq!(opts.assume_scope, AssumeScope::Real);
    assert_eq!(opts.budget.max_branches, 2);
    assert_eq!(opts.budget.max_depth, 3);
    assert!(opts.detailed_steps);
}

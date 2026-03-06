use crate::engine::Simplifier;
use crate::solve_runtime::SolveCtx;
use cas_ast::Equation;
use cas_solver_core::solve_analysis::PreflightContext;

pub(crate) fn build_solve_preflight_state(
    simplifier: &Simplifier,
    eq: &Equation,
    var: &str,
    value_domain: crate::ValueDomain,
    parent_ctx: &SolveCtx,
) -> PreflightContext<SolveCtx> {
    cas_solver_core::solve_runtime_flow::analyze_preflight_and_fork_context_with_existing_condition_derivation(
        &simplifier.context,
        eq,
        var,
        value_domain,
        parent_ctx,
        |expr, eval_domain| {
            crate::infer_implicit_domain(&simplifier.context, expr, eval_domain)
                .conditions()
                .iter()
                .cloned()
                .collect::<Vec<_>>()
        },
        crate::ImplicitDomain::empty,
        |domain, cond| {
            domain.conditions_mut().insert(cond);
        },
        |lhs, rhs, domain, eval_domain| {
            crate::derive_requires_from_equation(
                &simplifier.context,
                lhs,
                rhs,
                domain,
                eval_domain,
            )
        },
        crate::solve_runtime::SolveDomainEnv::new(),
        |domain_env, cond| {
            domain_env.required.conditions_mut().insert(cond.clone());
        },
    )
}

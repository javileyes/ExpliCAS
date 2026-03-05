use crate::engine::Simplifier;
use cas_ast::Equation;
use cas_solver_core::solve_analysis::{
    analyze_equation_preflight_and_fork_context_with,
    derive_equation_conditions_from_existing_with, PreflightContext,
};

use super::{SolveCtx, SolveDomainEnv};

pub(crate) type SolvePreflightState = PreflightContext<SolveCtx>;

/// Build per-level preflight state:
/// - domain exclusions from equation structure
/// - required conditions applied to domain env and shared sink
/// - child solve context with incremented depth
pub(crate) fn build_solve_preflight_state(
    simplifier: &Simplifier,
    eq: &Equation,
    var: &str,
    value_domain: crate::ValueDomain,
    parent_ctx: &SolveCtx,
) -> SolvePreflightState {
    analyze_equation_preflight_and_fork_context_with(
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
        |lhs, rhs, existing, eval_domain| {
            derive_equation_conditions_from_existing_with(
                lhs,
                rhs,
                existing,
                eval_domain,
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
            )
        },
        SolveDomainEnv::new(),
        |domain_env, cond| {
            domain_env.required.conditions_mut().insert(cond.clone());
        },
    )
}

//! Shared runtime helpers for solve preflight orchestration.

use cas_ast::{Equation, ExprId};
use std::hash::Hash;

/// Build solve preflight context using runtime-provided inference/derivation
/// callbacks and domain-environment constructors.
#[allow(clippy::too_many_arguments)]
pub fn build_preflight_context_with_existing_condition_derivation<
    C,
    V,
    Domain,
    DomainEnv,
    Assumption,
    Scope,
    FInferSide,
    FNewDomain,
    FInsertCondition,
    FDeriveFromDomain,
    FNewDomainEnv,
    FInsertRequiredIntoDomainEnv,
>(
    ctx: &cas_ast::Context,
    equation: &Equation,
    var: &str,
    value_domain: V,
    parent_ctx: &crate::solve_context::SolveContext<DomainEnv, C, Assumption, Scope>,
    infer_side_conditions: FInferSide,
    new_domain: FNewDomain,
    insert_condition: FInsertCondition,
    derive_from_domain: FDeriveFromDomain,
    new_domain_env: FNewDomainEnv,
    insert_required_into_domain_env: FInsertRequiredIntoDomainEnv,
) -> crate::solve_analysis::PreflightContext<
    crate::solve_context::SolveContext<DomainEnv, C, Assumption, Scope>,
>
where
    C: Eq + Hash + Clone,
    V: Copy,
    Assumption: Clone,
    Scope: Clone + PartialEq,
    FInferSide: FnMut(ExprId, V) -> Vec<C>,
    FNewDomain: FnMut() -> Domain,
    FInsertCondition: FnMut(&mut Domain, C),
    FDeriveFromDomain: FnMut(ExprId, ExprId, &Domain, V) -> Vec<C>,
    FNewDomainEnv: FnOnce() -> DomainEnv,
    FInsertRequiredIntoDomainEnv: FnMut(&mut DomainEnv, &C),
{
    crate::solve_runtime_flow::analyze_preflight_and_fork_context_with_existing_condition_derivation(
        ctx,
        equation,
        var,
        value_domain,
        parent_ctx,
        infer_side_conditions,
        new_domain,
        insert_condition,
        derive_from_domain,
        new_domain_env(),
        insert_required_into_domain_env,
    )
}

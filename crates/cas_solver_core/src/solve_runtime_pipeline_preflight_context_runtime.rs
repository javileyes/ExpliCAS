//! Shared runtime adapter for solve preflight state construction.

use cas_ast::{Equation, ExprId};
use std::hash::Hash;

/// Build preflight state from runtime state + context accessor and inference callbacks.
#[allow(clippy::too_many_arguments)]
pub fn build_solve_preflight_state_with_existing_condition_derivation_with_state<
    SState,
    C,
    V,
    Domain,
    DomainEnv,
    Assumption,
    Scope,
    FContextRef,
    FInferSide,
    FNewDomain,
    FInsertCondition,
    FDeriveFromDomain,
    FNewDomainEnv,
    FInsertRequiredIntoDomainEnv,
>(
    state: &SState,
    equation: &Equation,
    var: &str,
    value_domain: V,
    parent_ctx: &crate::solve_context::SolveContext<DomainEnv, C, Assumption, Scope>,
    context_ref: FContextRef,
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
    FContextRef: FnOnce(&SState) -> &cas_ast::Context,
    FInferSide: FnMut(ExprId, V) -> Vec<C>,
    FNewDomain: FnMut() -> Domain,
    FInsertCondition: FnMut(&mut Domain, C),
    FDeriveFromDomain: FnMut(ExprId, ExprId, &Domain, V) -> Vec<C>,
    FNewDomainEnv: FnOnce() -> DomainEnv,
    FInsertRequiredIntoDomainEnv: FnMut(&mut DomainEnv, &C),
{
    crate::solve_runtime_preflight_runtime::build_preflight_context_with_existing_condition_derivation(
        context_ref(state),
        equation,
        var,
        value_domain,
        parent_ctx,
        infer_side_conditions,
        new_domain,
        insert_condition,
        derive_from_domain,
        new_domain_env,
        insert_required_into_domain_env,
    )
}

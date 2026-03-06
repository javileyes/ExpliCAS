use cas_ast::{Equation, ExprId};
use std::hash::Hash;

/// Analyze solve preflight and derive equation-level conditions from an
/// existing-condition domain view.
///
/// This wraps:
/// - `analyze_equation_preflight_and_fork_context_with`
/// - `derive_equation_conditions_from_existing_with`
///
/// so runtime crates pass domain hooks once instead of wiring nested callbacks.
#[allow(clippy::too_many_arguments)]
pub fn analyze_preflight_and_fork_context_with_existing_condition_derivation<
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
    FInsertRequiredIntoDomainEnv,
>(
    ctx: &cas_ast::Context,
    equation: &Equation,
    var: &str,
    value_domain: V,
    parent_ctx: &crate::solve_context::SolveContext<DomainEnv, C, Assumption, Scope>,
    infer_side_conditions: FInferSide,
    mut new_domain: FNewDomain,
    mut insert_condition: FInsertCondition,
    mut derive_from_domain: FDeriveFromDomain,
    domain_env: DomainEnv,
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
    FInsertRequiredIntoDomainEnv: FnMut(&mut DomainEnv, &C),
{
    crate::solve_analysis::analyze_equation_preflight_and_fork_context_with(
        ctx,
        equation,
        var,
        value_domain,
        parent_ctx,
        infer_side_conditions,
        |lhs, rhs, existing, eval_domain| {
            crate::solve_analysis::derive_equation_conditions_from_existing_with(
                lhs,
                rhs,
                existing,
                eval_domain,
                &mut new_domain,
                &mut insert_condition,
                &mut derive_from_domain,
            )
        },
        domain_env,
        insert_required_into_domain_env,
    )
}

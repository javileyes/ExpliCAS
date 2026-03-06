//! Shared preflight-state construction bound to the current runtime solve
//! context/domain model.

use cas_ast::{Equation, ExprId};

/// Build a preflight context using the runtime solve context/domain aliases and
/// the standard implicit-domain accumulation policy.
pub fn build_runtime_solve_preflight_state_with_default_domain_inference_and_derivation<
    SState,
    FContextRef,
    FInferImplicitDomain,
    FDeriveRequires,
>(
    state: &SState,
    equation: &Equation,
    var: &str,
    value_domain: crate::value_domain::ValueDomain,
    parent_ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    context_ref: FContextRef,
    mut infer_implicit_domain: FInferImplicitDomain,
    mut derive_requires_from_equation: FDeriveRequires,
) -> crate::solve_analysis::PreflightContext<crate::solve_runtime_types::RuntimeSolveCtx>
where
    FContextRef: FnOnce(&SState) -> &cas_ast::Context,
    FInferImplicitDomain: FnMut(
        &cas_ast::Context,
        ExprId,
        crate::value_domain::ValueDomain,
    ) -> crate::solve_runtime_types::RuntimeImplicitDomain,
    FDeriveRequires: FnMut(
        &cas_ast::Context,
        ExprId,
        ExprId,
        &crate::solve_runtime_types::RuntimeImplicitDomain,
        crate::value_domain::ValueDomain,
    ) -> Vec<crate::solve_runtime_types::RuntimeImplicitCondition>,
{
    let core_ctx = context_ref(state);

    crate::solve_runtime_preflight_runtime::build_preflight_context_with_existing_condition_derivation(
        core_ctx,
        equation,
        var,
        value_domain,
        parent_ctx,
        |expr, eval_domain| {
            infer_implicit_domain(core_ctx, expr, eval_domain)
                .conditions()
                .iter()
                .cloned()
                .collect::<Vec<_>>()
        },
        crate::solve_runtime_types::RuntimeImplicitDomain::empty,
        |domain, cond| {
            domain.conditions_mut().insert(cond);
        },
        |lhs, rhs, domain, eval_domain| {
            derive_requires_from_equation(core_ctx, lhs, rhs, domain, eval_domain)
        },
        crate::solve_runtime_types::RuntimeSolveDomainEnv::new,
        |domain_env, cond| {
            domain_env.required.conditions_mut().insert(cond.clone());
        },
    )
}

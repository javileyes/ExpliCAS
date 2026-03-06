//! Shared equation-level additive cancellation bound to
//! [`RuntimeSolveAdapterState`].

use cas_ast::ExprId;

/// Semantic additive cancellation using the runtime-state simplifier contract.
pub fn cancel_additive_terms_semantic_with_runtime_state<S>(
    state: &mut S,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<crate::cancel_common_terms::CancelResult>
where
    S: crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
{
    crate::cancel_common_terms::cancel_additive_terms_semantic_runtime_with_state(
        state,
        lhs,
        rhs,
        |s| s.runtime_context(),
        |s| s.runtime_context_mut(),
        |s, term, opts| s.runtime_simplify_with_options_expr(term, opts),
    )
}

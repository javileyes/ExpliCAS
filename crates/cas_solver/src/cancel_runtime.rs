//! Equation-level additive cancellation runtime local to solver facade.

use cas_ast::{Context, ExprId};
use cas_solver_core::cancel_common_terms as cancel_common_terms_core;

/// Result of equation-level additive term cancellation.
pub type CancelResult = cancel_common_terms_core::CancelResult;

/// Cancel common additive terms between two expression trees.
pub fn cancel_common_additive_terms(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelResult> {
    cancel_common_terms_core::cancel_common_additive_terms(ctx, lhs, rhs)
}

/// Semantic fallback for equation-level term cancellation.
pub fn cancel_additive_terms_semantic(
    simplifier: &mut crate::Simplifier,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelResult> {
    cancel_common_terms_core::cancel_additive_terms_semantic_runtime_with_state(
        simplifier,
        lhs,
        rhs,
        |state| &state.context,
        |state| &mut state.context,
        |state, term, opts| state.simplify_with_stats(term, opts).0,
        crate::expand,
    )
}

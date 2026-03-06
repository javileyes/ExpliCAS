//! Equation-level additive cancellation runtime local to engine facade.

use cas_ast::ExprId;
use cas_solver_core::cancel_common_terms as cancel_common_terms_core;

/// Semantic fallback for equation-level term cancellation.
pub fn cancel_additive_terms_semantic(
    simplifier: &mut crate::Simplifier,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<cancel_common_terms_core::CancelResult> {
    cancel_common_terms_core::cancel_additive_terms_semantic_runtime_with_state(
        simplifier,
        lhs,
        rhs,
        |state| &state.context,
        |state| &mut state.context,
        |state, term, opts| state.simplify_with_stats(term, opts).0,
        crate::expand::expand,
    )
}

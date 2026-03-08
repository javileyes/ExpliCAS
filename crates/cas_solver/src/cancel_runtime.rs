//! Equation-level additive cancellation runtime local to solver facade.

use cas_ast::ExprId;
use cas_solver_core::cancel_common_terms as cancel_common_terms_core;

/// Semantic fallback for equation-level term cancellation.
pub fn cancel_additive_terms_semantic(
    simplifier: &mut crate::Simplifier,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<cancel_common_terms_core::CancelResult> {
    cas_solver_core::cancel_runtime_bound_runtime::cancel_additive_terms_semantic_with_runtime_state(
        simplifier, lhs, rhs,
    )
}

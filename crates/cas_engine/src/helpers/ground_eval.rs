//! Ground expression evaluation for predicate proofs.
//!
//! When `prove_nonzero` encounters a variable-free ("ground") expression it
//! can't structurally analyse (e.g. `cos(π/3)` or `2^(1/2) + 3^(1/2)`),
//! this module provides a fallback: clone the context into a lightweight
//! simplifier, simplify with `DomainMode::Generic`, and inspect the result.
//!
//! # Re-entrancy guard
//!
//! Because `simplify` itself may call `prove_nonzero` (via `has_undefined_risk`),
//! a `thread_local!` counter prevents infinite recursion.  When the counter is
//! non-zero, the fallback returns `None` immediately.

use cas_ast::{Context, ExprId};

use crate::Proof;

/// Attempt to prove non-zero by simplifying a ground expression.
///
/// Returns `Some(Proven)` if the expression simplifies to a non-zero `Number`,
/// `Some(Disproven)` if it simplifies to 0, and `None` if it can't determine
/// (expression doesn't fully reduce, or re-entrancy guard fires).
///
/// # Safety invariants
/// - Caller MUST ensure `!contains_variable(ctx, expr)`.
/// - Re-entrancy guard prevents `prove_nonzero → simplify → prove_nonzero` cycles.
pub(crate) fn try_ground_nonzero(ctx: &Context, expr: ExprId) -> Option<Proof> {
    cas_solver_core::predicate_proofs::try_ground_nonzero_with_shallow_recursive(
        ctx,
        expr,
        |source_ctx, source_expr| {
            let mut simplifier = crate::engine::Simplifier::with_context(source_ctx.clone());
            simplifier.set_collect_steps(false);
            let opts = crate::conservative_simplify::conservative_numeric_fold_options();

            let (result, _, _) = simplifier.simplify_with_stats(source_expr, opts);
            Some((simplifier.context, result))
        },
        try_ground_nonzero,
    )
}

#[cfg(test)]
mod tests;

//! Equation-level additive cancellation runtime local to solver facade.

use cas_ast::{Context, Expr, ExprId};
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
    use num_traits::Zero;

    let candidate_opts = crate::SimplifyOptions {
        collect_steps: false,
        ..Default::default()
    };

    let strict_proof_opts = crate::SimplifyOptions {
        shared: crate::SharedSemanticConfig {
            semantics: crate::EvalConfig::strict(),
            ..Default::default()
        },
        collect_steps: false,
        ..Default::default()
    };

    cancel_common_terms_core::cancel_additive_terms_semantic_with_state(
        simplifier,
        lhs,
        rhs,
        |state| &state.context,
        |state| &mut state.context,
        |state, term| state.simplify_with_stats(term, candidate_opts.clone()).0,
        crate::expand,
        |state, lt, rt| {
            let diff = state.context.add(Expr::Sub(lt, rt));
            let (simplified_diff, _, _) =
                state.simplify_with_stats(diff, strict_proof_opts.clone());
            matches!(state.context.get(simplified_diff), Expr::Number(n) if n.is_zero())
        },
    )
}

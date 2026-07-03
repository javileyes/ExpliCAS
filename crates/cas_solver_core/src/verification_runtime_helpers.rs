//! Shared runtime helpers for verification wrappers.
//!
//! This module centralizes small glue helpers used by runtime crates
//! (`cas_engine`, `cas_solver`) so they keep less duplicated orchestration code.

use cas_ast::{Context, ExprId};

/// Fold numeric islands under default limits with a caller-provided guard and
/// candidate evaluator.
pub(crate) fn fold_numeric_islands_with_guard_and_default_limits<
    Guard,
    FEnterGuard,
    FEvaluateCandidate,
>(
    ctx: &mut Context,
    root: ExprId,
    enter_guard: FEnterGuard,
    evaluate_candidate: FEvaluateCandidate,
) -> ExprId
where
    FEnterGuard: FnMut() -> Option<Guard>,
    FEvaluateCandidate: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
{
    crate::verification_numeric_islands::fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator(
        ctx,
        root,
        enter_guard,
        evaluate_candidate,
    )
}

/// Fold numeric islands using the standard ground-eval guard and default limits.
///
/// Runtime crates can provide only a candidate evaluator and avoid duplicating
/// the guard wiring.
pub(crate) fn fold_numeric_islands_with_default_guard_and_candidate<FEvaluateCandidate>(
    ctx: &mut Context,
    root: ExprId,
    evaluate_candidate: FEvaluateCandidate,
) -> ExprId
where
    FEvaluateCandidate: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
{
    fold_numeric_islands_with_guard_and_default_limits(
        ctx,
        root,
        cas_math::ground_eval_guard::GroundEvalGuard::enter,
        evaluate_candidate,
    )
}

/// Fold numeric islands using default guard/limits and the conservative
/// simplify-option preset.
///
/// The evaluator receives `(source_context, expr_id, conservative_options)`.
pub(crate) fn fold_numeric_islands_with_default_guard_and_conservative_options<FEvaluateCandidate>(
    ctx: &mut Context,
    root: ExprId,
    mut evaluate_candidate: FEvaluateCandidate,
) -> ExprId
where
    FEvaluateCandidate: FnMut(
        &Context,
        ExprId,
        &crate::simplify_options::SimplifyOptions,
    ) -> Option<(Context, ExprId)>,
{
    let fold_opts = crate::conservative_eval_config::conservative_numeric_fold_options();
    fold_numeric_islands_with_default_guard_and_candidate(ctx, root, |src_ctx, id| {
        evaluate_candidate(src_ctx, id, &fold_opts)
    })
}

#[cfg(test)]
mod tests {
    use super::fold_numeric_islands_with_guard_and_default_limits;

    struct DummyGuard;

    #[test]
    fn fold_wrapper_respects_guard_block() {
        let mut ctx = cas_ast::Context::new();
        let root = ctx.num(3);
        let out = fold_numeric_islands_with_guard_and_default_limits::<DummyGuard, _, _>(
            &mut ctx,
            root,
            || None,
            |_src_ctx, _id| None,
        );
        assert_eq!(out, root);
    }
}

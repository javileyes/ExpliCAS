//! Shared numeric-island verification helpers.
//!
//! Keeps guard + default-limits orchestration out of runtime crates.

use cas_ast::{Context, ExprId};

/// Fold numeric islands with default solver-core limits under a caller-provided
/// re-entrancy guard.
///
/// If `enter_guard()` returns `None`, the fold is skipped and `root` is
/// returned unchanged.
pub fn fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator<
    Guard,
    FEnterGuard,
    EvaluateCandidate,
>(
    ctx: &mut Context,
    root: ExprId,
    mut enter_guard: FEnterGuard,
    evaluate_candidate: EvaluateCandidate,
) -> ExprId
where
    FEnterGuard: FnMut() -> Option<Guard>,
    EvaluateCandidate: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
{
    let _guard = match enter_guard() {
        Some(guard) => guard,
        None => return root,
    };

    crate::numeric_islands::fold_numeric_islands_with_default_limits_and_candidate_evaluator(
        ctx,
        root,
        crate::verify_stats::record_skipped_limits,
        evaluate_candidate,
    )
}

#[cfg(test)]
mod tests {
    use super::fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator;

    struct DummyGuard;

    #[test]
    fn returns_root_when_guard_cannot_enter() {
        let mut ctx = cas_ast::Context::new();
        let root = ctx.num(2);
        let out = fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator::<
            DummyGuard,
            _,
            _,
        >(&mut ctx, root, || None, |_ctx, _id| None);
        assert_eq!(out, root);
    }

    #[test]
    fn executes_fold_path_when_guard_enters() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let root = ctx.add(cas_ast::Expr::Add(one, two));
        let out = fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator(
            &mut ctx,
            root,
            || Some(DummyGuard),
            |_src_ctx, _id| None,
        );
        // With evaluator disabled, fold is a no-op but should still return
        // a valid expression id from the same context.
        assert_eq!(out, root);
    }
}

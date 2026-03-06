//! Shared ground-evaluation adapters for runtime predicate/verification flows.

use cas_ast::{Context, ExprId};

pub(crate) fn ground_eval_candidate(
    source_ctx: &Context,
    source_expr: ExprId,
    opts: &cas_solver_core::simplify_options::SimplifyOptions,
) -> Option<(Context, ExprId)> {
    cas_solver_core::ground_eval_runtime::ground_eval_candidate_with_runtime_simplifier_with_state(
        source_ctx,
        source_expr,
        opts,
        crate::engine::Simplifier::with_context,
        |state, collect| state.set_collect_steps(collect),
        |state, expr, simplify_opts| state.simplify_with_stats(expr, simplify_opts).0,
        |state| state.context,
    )
}

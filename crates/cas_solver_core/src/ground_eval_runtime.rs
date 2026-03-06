//! Shared runtime helper to evaluate one candidate expression with a
//! host-provided simplifier state.

use cas_ast::{Context, ExprId};

/// Evaluate `source_expr` in a fresh runtime simplifier initialized from
/// `source_ctx`, with step collection disabled.
///
/// This keeps core logic independent from concrete simplifier types while
/// letting integration crates (`cas_engine`, `cas_solver`) provide adapters.
pub fn ground_eval_candidate_with_runtime_simplifier_with_state<
    SState,
    FBuildSimplifier,
    FSetCollectSteps,
    FSimplifyExprWithOptions,
    FIntoContext,
>(
    source_ctx: &Context,
    source_expr: ExprId,
    opts: &crate::simplify_options::SimplifyOptions,
    build_simplifier: FBuildSimplifier,
    mut set_collect_steps: FSetCollectSteps,
    mut simplify_expr_with_options: FSimplifyExprWithOptions,
    into_context: FIntoContext,
) -> Option<(Context, ExprId)>
where
    FBuildSimplifier: FnOnce(Context) -> SState,
    FSetCollectSteps: FnMut(&mut SState, bool),
    FSimplifyExprWithOptions:
        FnMut(&mut SState, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId,
    FIntoContext: FnOnce(SState) -> Context,
{
    let mut state = build_simplifier(source_ctx.clone());
    set_collect_steps(&mut state, false);
    let result = simplify_expr_with_options(&mut state, source_expr, opts.clone());
    let context = into_context(state);
    Some((context, result))
}

use crate::{Operation, PassStats};
use cas_ast::{Context, ExprId};

/// Expand with budget tracking, returning pass stats for charging.
pub fn expand(ctx: &mut Context, expr: ExprId) -> ExprId {
    cas_math::expand_ops::expand(ctx, expr)
}

/// Expand with budget tracking, returning pass stats for charging.
pub fn expand_with_stats(ctx: &mut Context, expr: ExprId) -> (ExprId, PassStats) {
    let nodes_snap = ctx.stats().nodes_created;
    let estimated_terms = cas_math::expand_estimate::estimate_expand_terms(ctx, expr).unwrap_or(0);
    let result = expand(ctx, expr);
    let nodes_delta = ctx.stats().nodes_created.saturating_sub(nodes_snap);

    let stats = PassStats {
        op: Operation::Expand,
        rewrite_count: 0,
        nodes_delta,
        terms_materialized: estimated_terms,
        poly_ops: 0,
        stop_reason: None,
    };

    (result, stats)
}

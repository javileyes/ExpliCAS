// =============================================================================
// Expression Complexity Helpers
// =============================================================================

use cas_ast::{Context, ExprId};

/// Count the number of nodes by **tree expansion** (no deduplication).
///
/// Shared DAG sub-expressions are counted once per reference.
/// This is the correct metric for the anti-worsen guard because it reflects
/// the work the simplifier does traversing the expression tree.
///
/// For guards on DAG-structured outputs (e.g., recurrence-built ASTs with heavy
/// sharing), use `count_nodes_dedup` in `inv_trig_n_angle.rs` instead.
///
/// Delegates to canonical `cas_ast::traversal::count_all_nodes`.
pub(crate) fn node_count_tree(ctx: &Context, expr: ExprId) -> usize {
    cas_ast::traversal::count_all_nodes(ctx, expr)
}

/// Check if a rewrite would "worsen" the expression by growing it too much.
/// Returns true if the rewrite should be BLOCKED.
///
/// Budget policy:
/// - Allow growth up to `max_growth_abs` nodes (e.g., 30)
/// - Allow growth up to `max_growth_ratio` times original size (e.g., 1.5x)
/// - If BOTH limits are exceeded, block the rewrite
pub(crate) fn rewrite_worsens_too_much(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    max_growth_abs: usize,
    max_growth_ratio: f64,
) -> bool {
    let size_before = node_count_tree(ctx, before);
    let size_after = node_count_tree(ctx, after);

    // If expression got smaller or stayed same, always allow
    if size_after <= size_before {
        return false;
    }

    let growth_abs = size_after - size_before;
    let growth_ratio = size_after as f64 / size_before.max(1) as f64;

    // Block only if BOTH thresholds are exceeded (more permissive)
    growth_abs > max_growth_abs && growth_ratio > max_growth_ratio
}

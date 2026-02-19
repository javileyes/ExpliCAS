//! Expression complexity helpers for rewrite guards.

use cas_ast::{Context, ExprId};

/// Count nodes by tree expansion (no DAG deduplication).
#[inline]
pub fn node_count_tree(ctx: &Context, expr: ExprId) -> usize {
    cas_ast::traversal::count_all_nodes(ctx, expr)
}

/// Check whether a rewrite grows expression size beyond guard thresholds.
///
/// Returns `true` when both absolute and ratio growth exceed the provided
/// limits.
pub fn rewrite_worsens_too_much(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    max_growth_abs: usize,
    max_growth_ratio: f64,
) -> bool {
    let size_before = node_count_tree(ctx, before);
    let size_after = node_count_tree(ctx, after);

    if size_after <= size_before {
        return false;
    }

    let growth_abs = size_after - size_before;
    let growth_ratio = size_after as f64 / size_before.max(1) as f64;
    growth_abs > max_growth_abs && growth_ratio > max_growth_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn node_count_tree_counts_nodes() {
        let mut ctx = Context::new();
        let expr = parse("(a+b)*(c+d)", &mut ctx).expect("parse");
        assert!(node_count_tree(&ctx, expr) >= 7);
    }

    #[test]
    fn worsen_guard_triggers_only_when_both_thresholds_exceeded() {
        let mut ctx = Context::new();
        let before = parse("x", &mut ctx).expect("parse");
        let after = parse("(x+1)*(x+2)*(x+3)", &mut ctx).expect("parse");

        assert!(rewrite_worsens_too_much(&ctx, before, after, 3, 1.5));
        assert!(!rewrite_worsens_too_much(&ctx, before, after, 30, 1.5));
        assert!(!rewrite_worsens_too_much(&ctx, before, after, 3, 100.0));
    }
}

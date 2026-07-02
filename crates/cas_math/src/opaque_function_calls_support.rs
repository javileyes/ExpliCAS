//! Helpers for collecting and pairing function-call subexpressions.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, ExprId};
use std::cmp::Ordering;

/// Greedy one-to-one structural matching between left/right function call lists.
pub(crate) fn match_shared_calls_structural(
    ctx: &Context,
    left: &[ExprId],
    right: &[ExprId],
    max_pairs: usize,
) -> Vec<(ExprId, ExprId)> {
    let mut shared: Vec<(ExprId, ExprId)> = Vec::new();
    let mut used_right = vec![false; right.len()];
    for &lc in left {
        for (j, &rc) in right.iter().enumerate() {
            if !used_right[j] && compare_expr(ctx, lc, rc) == Ordering::Equal {
                shared.push((lc, rc));
                used_right[j] = true;
                break;
            }
        }
        if shared.len() >= max_pairs {
            break;
        }
    }
    shared
}

#[cfg(test)]
mod tests {}

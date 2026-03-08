//! Helpers for collecting and pairing function-call subexpressions.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

/// Collect `Expr::Function` nodes up to `max_depth`.
pub fn collect_function_calls_limited(
    ctx: &Context,
    expr: ExprId,
    max_depth: usize,
) -> Vec<ExprId> {
    let mut out = Vec::new();
    collect_calls_recursive(ctx, expr, &mut out, 0, max_depth);
    out
}

fn collect_calls_recursive(
    ctx: &Context,
    expr: ExprId,
    out: &mut Vec<ExprId>,
    depth: usize,
    max_depth: usize,
) {
    if depth > max_depth {
        return;
    }
    match ctx.get(expr) {
        Expr::Function(_, _) => out.push(expr),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_calls_recursive(ctx, *l, out, depth + 1, max_depth);
            collect_calls_recursive(ctx, *r, out, depth + 1, max_depth);
        }
        Expr::Pow(base, exp) => {
            collect_calls_recursive(ctx, *base, out, depth + 1, max_depth);
            collect_calls_recursive(ctx, *exp, out, depth + 1, max_depth);
        }
        Expr::Neg(inner) => collect_calls_recursive(ctx, *inner, out, depth + 1, max_depth),
        _ => {}
    }
}

/// Greedy one-to-one structural matching between left/right function call lists.
pub fn match_shared_calls_structural(
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
mod tests {
    use super::{collect_function_calls_limited, match_shared_calls_structural};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn collects_calls_with_depth_limit() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)+cos(y)", &mut ctx).expect("parse");
        let calls = collect_function_calls_limited(&ctx, expr, 4);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn matches_shared_calls_structurally() {
        let mut ctx = Context::new();
        let left_expr = parse("sin(x)+cos(y)", &mut ctx).expect("parse left");
        let right_expr = parse("cos(y)+tan(z)", &mut ctx).expect("parse right");
        let left = collect_function_calls_limited(&ctx, left_expr, 4);
        let right = collect_function_calls_limited(&ctx, right_expr, 4);
        let shared = match_shared_calls_structural(&ctx, &left, &right, 3);
        assert_eq!(shared.len(), 1);
    }
}

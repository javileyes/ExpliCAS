//! Normal-form scoring helpers used to compare rewrite quality.

use cas_ast::{Context, Expr, ExprId};

/// Count total nodes in an expression tree.
#[inline]
pub fn count_all_nodes(ctx: &Context, expr: ExprId) -> usize {
    cas_ast::traversal::count_all_nodes(ctx, expr)
}

/// Compare NF score lazily; returns true when `after` is strictly better.
pub fn nf_score_after_is_better(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let before_base = nf_score_base(ctx, before);
    let after_base = nf_score_base(ctx, after);

    if after_base < before_base {
        return true;
    }
    if after_base > before_base {
        return false;
    }

    let before_inv = mul_unsorted_adjacent(ctx, before);
    let after_inv = mul_unsorted_adjacent(ctx, after);
    after_inv < before_inv
}

fn nf_score_base(ctx: &Context, id: ExprId) -> (usize, usize) {
    let mut divs_subs = 0;
    let mut total = 0;
    let mut stack = vec![id];

    while let Some(node_id) = stack.pop() {
        total += 1;

        match ctx.get(node_id) {
            Expr::Div(..) | Expr::Sub(..) => divs_subs += 1,
            _ => {}
        }

        match ctx.get(node_id) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    (divs_subs, total)
}

/// Count out-of-order adjacent pairs in right-associated `Mul` chains.
pub fn mul_unsorted_adjacent(ctx: &Context, root: ExprId) -> usize {
    use cas_ast::ordering::compare_expr;
    use std::cmp::Ordering;
    use std::collections::HashSet;

    let mut mul_nodes: HashSet<ExprId> = HashSet::new();
    let mut mul_right_children: HashSet<ExprId> = HashSet::new();

    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                mul_nodes.insert(id);
                if matches!(ctx.get(*r), Expr::Mul(..)) {
                    mul_right_children.insert(*r);
                }
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    let heads: Vec<_> = mul_nodes.difference(&mul_right_children).copied().collect();
    let mut inversions = 0;

    for head in heads {
        let mut factors = Vec::new();
        let mut current = head;

        loop {
            if let Expr::Mul(l, r) = ctx.get(current).clone() {
                factors.push(l);
                if matches!(ctx.get(r), Expr::Mul(..)) {
                    current = r;
                } else {
                    factors.push(r);
                    break;
                }
            } else {
                factors.push(current);
                break;
            }
        }

        for pair in factors.windows(2) {
            if compare_expr(ctx, pair[0], pair[1]) == Ordering::Greater {
                inversions += 1;
            }
        }
    }

    inversions
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use cas_parser::parse;

    #[test]
    fn mul_unsorted_detects_inversions() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let z = ctx.var("z");

        // Build a right-associated sorted chain: a * (b * z)
        let bz = ctx.add_raw(Expr::Mul(b, z));
        let sorted = ctx.add_raw(Expr::Mul(a, bz));

        // Build a right-associated unsorted chain: a * (z * b)
        let zb = ctx.add_raw(Expr::Mul(z, b));
        let unsorted = ctx.add_raw(Expr::Mul(a, zb));

        assert!(mul_unsorted_adjacent(&ctx, unsorted) > mul_unsorted_adjacent(&ctx, sorted));
    }

    #[test]
    fn nf_score_prefers_fewer_divs_subs() {
        let mut ctx = Context::new();
        let before = parse("a/b", &mut ctx).expect("parse");
        let after = parse("a*b", &mut ctx).expect("parse");

        assert!(nf_score_after_is_better(&ctx, before, after));
    }
}

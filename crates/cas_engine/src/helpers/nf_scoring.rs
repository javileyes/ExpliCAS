// ========== Normal Form Scoring ==========

use cas_ast::{Context, Expr, ExprId};

/// Count total nodes in an expression tree.
///
/// Delegates to canonical `cas_ast::traversal::count_all_nodes`.
pub fn count_all_nodes(ctx: &Context, expr: ExprId) -> usize {
    cas_ast::traversal::count_all_nodes(ctx, expr)
}

/// Count nodes matching a predicate.
///
/// Wrapper calling canonical `cas_ast::traversal::count_nodes_matching`.
/// (See POLICY.md "Traversal Contract")
pub fn count_nodes_matching<F>(ctx: &Context, expr: ExprId, pred: F) -> usize
where
    F: FnMut(&Expr) -> bool,
{
    cas_ast::traversal::count_nodes_matching(ctx, expr, pred)
}

/// Score expression for normal form quality (lower is better).
/// Returns (divs_subs, total_nodes, mul_inversions) for lexicographic comparison.
///
/// Expressions with fewer Div/Sub nodes are preferred (C2 canonical form).
/// Ties are broken by total node count (simpler is better).
/// Final tie-breaker: fewer out-of-order adjacent pairs in Mul chains.
///
/// For performance-critical comparisons, use `compare_nf_score_lazy` instead.
pub fn nf_score(ctx: &Context, id: ExprId) -> (usize, usize, usize) {
    let divs_subs = count_nodes_matching(ctx, id, |e| matches!(e, Expr::Div(..) | Expr::Sub(..)));
    let total = count_all_nodes(ctx, id);
    let inversions = mul_unsorted_adjacent(ctx, id);
    (divs_subs, total, inversions)
}

/// First two components of nf_score: (divs_subs, total_nodes)
/// Uses single traversal for efficiency (counts both in one pass).
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

        // Push children
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
            // Leaves
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    (divs_subs, total)
}

/// Compare nf_score lazily: only computes mul_unsorted_adjacent if first two components tie.
/// Returns true if `after` is strictly better (lower) than `before`.
pub fn nf_score_after_is_better(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let before_base = nf_score_base(ctx, before);
    let after_base = nf_score_base(ctx, after);

    // Compare first two components
    if after_base < before_base {
        return true; // Clear improvement
    }
    if after_base > before_base {
        return false; // Worse
    }

    // Tie on (divs_subs, total) - need to compare mul_inversions
    let before_inv = mul_unsorted_adjacent(ctx, before);
    let after_inv = mul_unsorted_adjacent(ctx, after);
    after_inv < before_inv
}

/// Count out-of-order adjacent pairs in Mul chains (right-associative).
///
/// For a chain `a * (b * (c * d))` with factors `[a, b, c, d]`:
/// - Counts how many pairs (f[i], f[i+1]) have compare_expr(f[i], f[i+1]) == Greater
///
/// This metric allows canonicalizing rewrites that only reorder Mul factors.
pub fn mul_unsorted_adjacent(ctx: &Context, root: ExprId) -> usize {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;
    use std::collections::HashSet;

    // Collect all Mul nodes and identify which are right-children of other Muls
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
            // Leaves
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    // Heads are Mul nodes that are NOT the right child of another Mul
    let heads: Vec<_> = mul_nodes.difference(&mul_right_children).copied().collect();

    let mut inversions = 0;

    for head in heads {
        // Linearize factors by following right-assoc pattern: a*(b*(c*d)) -> [a,b,c,d]
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

        // Count adjacent inversions
        for pair in factors.windows(2) {
            if compare_expr(ctx, pair[0], pair[1]) == Ordering::Greater {
                inversions += 1;
            }
        }
    }

    inversions
}

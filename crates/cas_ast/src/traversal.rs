//! Canonical AST traversal utilities.
//!
//! This module provides stack-safe, iterative traversal functions for expression trees.
//! All callsites should use these instead of implementing their own recursive traversals.
//!
//! # Functions
//!
//! - [`count_all_nodes`]: Count total nodes in a tree
//! - [`count_nodes_matching`]: Count nodes matching a predicate
//! - [`count_nodes_and_max_depth`]: Get both node count and max depth
//! - [`collect_variables`]: Collect all unique variable names
//! - [`substitute_expr_by_id`]: Replace a subtree by ExprId
//!
//! # Why Iterative?
//!
//! Recursive traversal can overflow the stack on very deep expressions.
//! These functions use explicit stacks, making them safe for any tree depth.
//!
//! # See Also
//!
//! - POLICY.md "Traversal Contract" for contribution rules

use crate::expression::{Context, Expr, ExprId};
use std::collections::HashSet;

/// Count all nodes in an expression tree.
///
/// **CANONICAL traversal function for simple counting.**
///
/// Stack-safe (iterative implementation).
///
/// # Example
/// ```ignore
/// let total = count_all_nodes(&ctx, root);
/// ```
pub fn count_all_nodes(ctx: &Context, root: ExprId) -> usize {
    count_nodes_matching(ctx, root, |_| true)
}

/// Count nodes matching a predicate.
///
/// **CANONICAL traversal function for filtered counting.**
///
/// Stack-safe (iterative implementation using explicit stack).
///
/// # Arguments
/// - `ctx`: The expression context
/// - `root`: Root expression to traverse
/// - `pred`: Predicate function, returns `true` for nodes to count
///
/// # Example
/// ```ignore
/// // Count only Div nodes
/// let div_count = count_nodes_matching(&ctx, root, |e| matches!(e, Expr::Div(_, _)));
/// ```
pub fn count_nodes_matching<F>(ctx: &Context, root: ExprId, mut pred: F) -> usize
where
    F: FnMut(&Expr) -> bool,
{
    let mut count = 0;
    let mut stack = vec![root];

    while let Some(id) = stack.pop() {
        let node = ctx.get(id);
        if pred(node) {
            count += 1;
        }
        push_children(node, &mut stack);
    }

    count
}

/// Count nodes and compute maximum depth.
///
/// **CANONICAL traversal function for complexity metrics.**
///
/// Stack-safe (iterative implementation using explicit stack with depth tracking).
///
/// # Returns
/// Tuple of (total_nodes, max_depth) where:
/// - `total_nodes`: Total count of all nodes
/// - `max_depth`: Maximum distance from root to any leaf (root has depth 0)
///
/// # Example
/// ```ignore
/// let (nodes, depth) = count_nodes_and_max_depth(&ctx, root);
/// if depth > 100 { println!("Very deep expression!"); }
/// ```
pub fn count_nodes_and_max_depth(ctx: &Context, root: ExprId) -> (usize, usize) {
    let mut count = 0;
    let mut max_depth = 0;
    let mut stack: Vec<(ExprId, usize)> = vec![(root, 0)];

    while let Some((id, depth)) = stack.pop() {
        count += 1;
        max_depth = max_depth.max(depth);

        let node = ctx.get(id);
        let child_depth = depth + 1;

        match node {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push((*l, child_depth));
                stack.push((*r, child_depth));
            }
            Expr::Neg(e) | Expr::Hold(e) => stack.push((*e, child_depth)),
            Expr::Function(_, args) => {
                for &arg in args {
                    stack.push((arg, child_depth));
                }
            }
            Expr::Matrix { data, .. } => {
                for &elem in data {
                    stack.push((elem, child_depth));
                }
            }
            // Leaves — no children
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    (count, max_depth)
}

/// Helper: push all children of a node onto the stack.
#[inline]
fn push_children(node: &Expr, stack: &mut Vec<ExprId>) {
    match node {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            stack.push(*l);
            stack.push(*r);
        }
        Expr::Neg(e) | Expr::Hold(e) => stack.push(*e),
        Expr::Function(_, args) => stack.extend(args),
        Expr::Matrix { data, .. } => stack.extend(data),
        // Leaves — no children
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}

/// Collect all unique variable names in an expression tree.
///
/// **CANONICAL traversal function for variable discovery.**
///
/// Stack-safe (iterative implementation using explicit stack).
///
/// # Example
/// ```ignore
/// let vars = collect_variables(&ctx, root);
/// assert!(vars.contains("x"));
/// ```
pub fn collect_variables(ctx: &Context, root: ExprId) -> HashSet<String> {
    let mut vars = HashSet::new();
    let mut stack = vec![root];

    while let Some(id) = stack.pop() {
        let node = ctx.get(id);
        if let Expr::Variable(sym_id) = node {
            vars.insert(ctx.sym_name(*sym_id).to_string());
        }
        push_children(node, &mut stack);
    }

    vars
}

/// Substitute occurrences of `target` with `replacement` anywhere in the expression tree.
///
/// **CANONICAL traversal function for id-based substitution.**
///
/// Returns a new `ExprId` if any substitution occurred, otherwise returns the original `root`.
/// The match uses exhaustive pattern matching on `Expr` variants so the compiler will
/// force an update here when new variants are added.
///
/// # Example
/// ```ignore
/// let new_root = substitute_expr_by_id(&mut ctx, root, old_id, new_id);
/// ```
pub fn substitute_expr_by_id(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
) -> ExprId {
    if root == target {
        return replacement;
    }

    let expr = ctx.get(root).clone();
    match expr {
        Expr::Add(l, r) => {
            let new_l = substitute_expr_by_id(ctx, l, target, replacement);
            let new_r = substitute_expr_by_id(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                ctx.add(Expr::Add(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Sub(l, r) => {
            let new_l = substitute_expr_by_id(ctx, l, target, replacement);
            let new_r = substitute_expr_by_id(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                ctx.add(Expr::Sub(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_expr_by_id(ctx, l, target, replacement);
            let new_r = substitute_expr_by_id(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                ctx.add(Expr::Mul(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Div(l, r) => {
            let new_l = substitute_expr_by_id(ctx, l, target, replacement);
            let new_r = substitute_expr_by_id(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                ctx.add(Expr::Div(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Pow(b, e) => {
            let new_b = substitute_expr_by_id(ctx, b, target, replacement);
            let new_e = substitute_expr_by_id(ctx, e, target, replacement);
            if new_b != b || new_e != e {
                ctx.add(Expr::Pow(new_b, new_e))
            } else {
                root
            }
        }
        Expr::Neg(inner) => {
            let new_inner = substitute_expr_by_id(ctx, inner, target, replacement);
            if new_inner != inner {
                ctx.add(Expr::Neg(new_inner))
            } else {
                root
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args.iter() {
                let new_arg = substitute_expr_by_id(ctx, *arg, target, replacement);
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                root
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut new_data = Vec::new();
            let mut changed = false;
            for elem in data.iter() {
                let new_elem = substitute_expr_by_id(ctx, *elem, target, replacement);
                if new_elem != *elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                root
            }
        }
        Expr::Hold(inner) => {
            let new_inner = substitute_expr_by_id(ctx, inner, target, replacement);
            if new_inner != inner {
                ctx.add(Expr::Hold(new_inner))
            } else {
                root
            }
        }
        // Leaves — no children to recurse into.
        // Explicit listing so `rustc` catches missing variants when Expr grows.
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => root,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::Context;

    #[test]
    fn test_count_all_nodes_simple() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        // x + y = 3 nodes (Add, x, y)
        let sum = ctx.add_raw(Expr::Add(x, y));

        assert_eq!(count_all_nodes(&ctx, sum), 3);
    }

    #[test]
    fn test_count_all_nodes_nested() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        // (a + b) * c = 5 nodes
        let ab = ctx.add_raw(Expr::Add(a, b));
        let abc = ctx.add_raw(Expr::Mul(ab, c));

        assert_eq!(count_all_nodes(&ctx, abc), 5);
    }

    #[test]
    fn test_count_nodes_matching_div() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        // x / y + z = 1 Div node
        let xy = ctx.add_raw(Expr::Div(x, y));
        let expr = ctx.add_raw(Expr::Add(xy, z));

        let div_count = count_nodes_matching(&ctx, expr, |e| matches!(e, Expr::Div(_, _)));
        assert_eq!(div_count, 1);
    }

    #[test]
    fn test_count_nodes_and_max_depth() {
        let mut ctx = Context::new();
        // Build a chain: a + (b + (c + d)) = depth 3
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let d = ctx.var("d");
        let cd = ctx.add_raw(Expr::Add(c, d)); // depth 2
        let bcd = ctx.add_raw(Expr::Add(b, cd)); // depth 1
        let abcd = ctx.add_raw(Expr::Add(a, bcd)); // depth 0

        let (nodes, depth) = count_nodes_and_max_depth(&ctx, abcd);
        assert_eq!(nodes, 7); // 3 Add + 4 vars
        assert_eq!(depth, 3);
    }

    #[test]
    fn test_deep_tree_no_stack_overflow() {
        let mut ctx = Context::new();

        // Build a deep chain using unique variables: Add(x1, Add(x2, Add(x3, ...)))
        let mut curr = ctx.var("x0");
        for i in 1..=100 {
            let new_var = ctx.var(&format!("x{}", i));
            curr = ctx.add_raw(Expr::Add(new_var, curr));
        }

        // Should not overflow - iterative implementation
        let (nodes, depth) = count_nodes_and_max_depth(&ctx, curr);
        // 100 Add nodes + 101 variables = 201 nodes
        assert_eq!(nodes, 201);
        assert_eq!(depth, 100);
    }

    #[test]
    fn test_collect_variables_basic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.add_raw(Expr::Number(num_rational::BigRational::from_integer(
            2.into(),
        )));
        // 2*x + y
        let mul = ctx.add_raw(Expr::Mul(two, x));
        let expr = ctx.add_raw(Expr::Add(mul, y));

        let vars = collect_variables(&ctx, expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_collect_variables_no_duplicates() {
        let mut ctx = Context::new();
        let x1 = ctx.var("x");
        let x2 = ctx.var("x");
        // x + x — should yield {"x"}, not 2 entries
        let expr = ctx.add_raw(Expr::Add(x1, x2));

        let vars = collect_variables(&ctx, expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_collect_variables_constant_only() {
        let mut ctx = Context::new();
        let n = ctx.add_raw(Expr::Number(num_rational::BigRational::from_integer(
            42.into(),
        )));
        let vars = collect_variables(&ctx, n);
        assert!(vars.is_empty());
    }
}

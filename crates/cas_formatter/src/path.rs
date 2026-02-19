//! Expression-path helpers for presentation-layer subtree highlighting.

use cas_ast::{Context, Expr, ExprId, ExprPath};

/// Extract terms from an Add/Sub chain for multi-term highlighting (V2.9.17)
/// Returns individual ExprIds that make up the chain.
/// Also unwraps Neg wrappers since they may be dynamically created and not exist
/// in the original expression tree.
pub fn extract_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    fn collect_terms(ctx: &Context, id: ExprId, terms: &mut Vec<ExprId>) {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                collect_terms(ctx, *l, terms);
                collect_terms(ctx, *r, terms);
            }
            Expr::Neg(inner) => {
                // Unwrap Neg to find the underlying term (which may exist in original tree)
                // Add both: the Neg itself (if it exists) and the inner term
                terms.push(id); // The Neg wrapper
                terms.push(*inner); // The inner term (more likely to exist in original)
            }
            _ => terms.push(id),
        }
    }
    collect_terms(ctx, expr, &mut terms);
    terms
}

/// Diff-based fallback: find path to target expression within a tree (V2.9.18)
/// When direct path lookup fails, this performs a depth-first search to find
/// where the target expression appears in the tree.
/// Returns the path if found, or None if not found.
pub fn diff_find_path_to_expr(ctx: &Context, root: ExprId, target: ExprId) -> Option<ExprPath> {
    // First, try direct ExprId match
    if root == target {
        return Some(vec![]);
    }

    // Recursively search children
    fn search(ctx: &Context, current: ExprId, target: ExprId, path: &mut ExprPath) -> bool {
        if current == target {
            return true;
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                // Try left branch
                path.push(0); // Left = 0
                if search(ctx, *l, target, path) {
                    return true;
                }
                path.pop();

                // Try right branch
                path.push(1); // Right = 1
                if search(ctx, *r, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Neg(inner) => {
                path.push(0);
                if search(ctx, *inner, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Function(_, args) => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i as u8);
                    if search(ctx, *arg, target, path) {
                        return true;
                    }
                    path.pop();
                }
            }
            Expr::Hold(inner) => {
                path.push(0);
                if search(ctx, *inner, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Matrix { data, .. } => {
                for (i, elem) in data.iter().enumerate() {
                    path.push(i as u8);
                    if search(ctx, *elem, target, path) {
                        return true;
                    }
                    path.pop();
                }
            }
            // Leaves
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
        false
    }

    let mut path = vec![];
    if search(ctx, root, target, &mut path) {
        Some(path)
    } else {
        None
    }
}

/// Find ALL paths to a target expression within a tree (V2.9.19)
/// Unlike diff_find_path_to_expr which returns only the first match,
/// this finds every occurrence (useful for x+x where both x share same ExprId in DAG)
pub fn diff_find_all_paths_to_expr(ctx: &Context, root: ExprId, target: ExprId) -> Vec<ExprPath> {
    let mut results = Vec::new();

    fn search(
        ctx: &Context,
        current: ExprId,
        target: ExprId,
        path: &mut ExprPath,
        results: &mut Vec<ExprPath>,
    ) {
        if current == target {
            results.push(path.clone());
            // Continue searching - there may be more occurrences below or in siblings
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                // Search left branch
                path.push(0);
                search(ctx, *l, target, path, results);
                path.pop();

                // Search right branch
                path.push(1);
                search(ctx, *r, target, path, results);
                path.pop();
            }
            Expr::Neg(inner) => {
                path.push(0);
                search(ctx, *inner, target, path, results);
                path.pop();
            }
            Expr::Function(_, args) => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i as u8);
                    search(ctx, *arg, target, path, results);
                    path.pop();
                }
            }
            Expr::Hold(inner) => {
                path.push(0);
                search(ctx, *inner, target, path, results);
                path.pop();
            }
            Expr::Matrix { data, .. } => {
                for (i, elem) in data.iter().enumerate() {
                    path.push(i as u8);
                    search(ctx, *elem, target, path, results);
                    path.pop();
                }
            }
            // Leaves
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    let mut path = vec![];
    search(ctx, root, target, &mut path, &mut results);
    results
}

/// Find paths to a target expression by structural equivalence (V2.9.24)
/// Unlike diff_find_all_paths_to_expr which matches by ExprId,
/// this uses compare_expr for structural comparison.
/// Essential for dynamically constructed before_local expressions
/// where terms may have different ExprIds than the original tree.
pub fn diff_find_paths_by_structure(ctx: &Context, root: ExprId, target: ExprId) -> Vec<ExprPath> {
    let mut results = Vec::new();

    fn search(
        ctx: &Context,
        current: ExprId,
        target: ExprId,
        path: &mut ExprPath,
        results: &mut Vec<ExprPath>,
    ) {
        // Check structural equivalence using compare_expr
        if cas_ast::ordering::compare_expr(ctx, current, target) == std::cmp::Ordering::Equal {
            results.push(path.clone());
            // Don't recurse into children if we matched the whole subtree
            return;
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                // Search left branch
                path.push(0);
                search(ctx, *l, target, path, results);
                path.pop();

                // Search right branch
                path.push(1);
                search(ctx, *r, target, path, results);
                path.pop();
            }
            Expr::Neg(inner) => {
                path.push(0);
                search(ctx, *inner, target, path, results);
                path.pop();
            }
            Expr::Function(_, args) => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i as u8);
                    search(ctx, *arg, target, path, results);
                    path.pop();
                }
            }
            Expr::Hold(inner) => {
                path.push(0);
                search(ctx, *inner, target, path, results);
                path.pop();
            }
            Expr::Matrix { data, .. } => {
                for (i, elem) in data.iter().enumerate() {
                    path.push(i as u8);
                    search(ctx, *elem, target, path, results);
                    path.pop();
                }
            }
            // Leaves
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    let mut path = vec![];
    search(ctx, root, target, &mut path, &mut results);
    results
}

/// Navigate to the subexpression at a given path within an expression tree
pub fn navigate_to_subexpr(ctx: &Context, mut current: ExprId, path: &ExprPath) -> ExprId {
    for &step in path {
        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                current = if step == 0 { *l } else { *r };
            }
            Expr::Neg(inner) => {
                current = *inner;
            }
            Expr::Function(_, args) => {
                if let Some(&arg) = args.get(step as usize) {
                    current = arg;
                }
            }
            Expr::Hold(inner) => {
                current = *inner;
            }
            Expr::Matrix { data, .. } => {
                if let Some(&elem) = data.get(step as usize) {
                    current = elem;
                } else {
                    break;
                }
            }
            // Leaves
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => break,
        }
    }
    current
}

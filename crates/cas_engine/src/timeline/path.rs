//! Compatibility shim for timeline path helpers.
//! Most path search/navigation utilities now live in `cas_formatter::path`.

use crate::step::PathStep;
use cas_ast::{Context, Expr, ExprId};

pub(super) use cas_formatter::path::{
    diff_find_all_paths_to_expr, diff_find_path_to_expr, diff_find_paths_by_structure,
    extract_add_terms, navigate_to_subexpr,
};

/// Convert PathStep to u8 for ExprPath.
#[inline]
pub(super) fn pathstep_to_u8(ps: &PathStep) -> u8 {
    ps.to_child_index()
}

/// Find path from root to target expression.
/// Returns empty Vec if target not found or target == root.
pub(super) fn find_path_to_expr(ctx: &Context, root: ExprId, target: ExprId) -> Vec<PathStep> {
    if root == target {
        return vec![];
    }

    // DFS to find target within root.
    fn dfs(ctx: &Context, current: ExprId, target: ExprId, path: &mut Vec<PathStep>) -> bool {
        if current == target {
            return true;
        }

        match ctx.get(current) {
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                path.push(PathStep::Left);
                if dfs(ctx, *l, target, path) {
                    return true;
                }
                path.pop();

                path.push(PathStep::Right);
                if dfs(ctx, *r, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Pow(base, exp) => {
                path.push(PathStep::Base);
                if dfs(ctx, *base, target, path) {
                    return true;
                }
                path.pop();

                path.push(PathStep::Exponent);
                if dfs(ctx, *exp, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Neg(inner) => {
                path.push(PathStep::Inner);
                if dfs(ctx, *inner, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Function(_, args) => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(PathStep::Arg(i));
                    if dfs(ctx, *arg, target, path) {
                        return true;
                    }
                    path.pop();
                }
            }
            Expr::Hold(inner) => {
                path.push(PathStep::Inner);
                if dfs(ctx, *inner, target, path) {
                    return true;
                }
                path.pop();
            }
            Expr::Matrix { data, .. } => {
                for (i, elem) in data.iter().enumerate() {
                    path.push(PathStep::Arg(i));
                    if dfs(ctx, *elem, target, path) {
                        return true;
                    }
                    path.pop();
                }
            }
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }

        false
    }

    let mut path = Vec::new();
    if dfs(ctx, root, target, &mut path) {
        path
    } else {
        vec![]
    }
}

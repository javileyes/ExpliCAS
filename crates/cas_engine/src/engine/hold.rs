//! Hold semantics and expression substitution utilities.
//!
//! Functions for managing HoldAll function semantics, unwrapping hold barriers,
//! and performing structural expression substitution by ExprId.

use cas_ast::{Context, Expr, ExprId};

// =============================================================================
// HoldAll function semantics
// =============================================================================

/// Returns true if a function has HoldAll semantics, meaning its arguments
/// should NOT be simplified before the function rule is applied.
/// This is crucial for functions like poly_gcd that need to see the raw
/// multiplicative structure of their arguments.
/// Also includes hold function which is an internal invisible barrier.
pub(super) fn is_hold_all_function(name: &str) -> bool {
    matches!(name, "poly_gcd" | "pgcd") || cas_ast::hold::is_hold_name(name)
}

/// Unwrap top-level __hold() wrapper after simplification.
/// This is called at the end of eval/simplify so the user sees clean results
/// without the INTERNAL barrier visible (user-facing hold() is preserved).
pub(super) fn unwrap_hold_top(ctx: &Context, expr: ExprId) -> ExprId {
    cas_ast::hold::unwrap_internal_hold(ctx, expr)
}

/// Re-export strip_all_holds from cas_ast for use by rules.
///
/// This is the CANONICAL implementation - see cas_ast::hold for the contract.
/// Do NOT duplicate this function elsewhere.
pub fn strip_all_holds(ctx: &mut Context, expr: ExprId) -> ExprId {
    cas_ast::hold::strip_all_holds(ctx, expr)
}

/// Substitute occurrences of `target` with `replacement` anywhere in the expression tree.
/// Returns new ExprId if substitution occurred, otherwise returns original root.
pub fn substitute_expr_by_id(
    context: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
) -> ExprId {
    if root == target {
        return replacement;
    }

    let expr = context.get(root).clone();
    match expr {
        Expr::Add(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l != l || new_r != r {
                context.add(Expr::Add(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Sub(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l != l || new_r != r {
                context.add(Expr::Sub(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l != l || new_r != r {
                context.add(Expr::Mul(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Div(l, r) => {
            let new_l = substitute_expr_by_id(context, l, target, replacement);
            let new_r = substitute_expr_by_id(context, r, target, replacement);
            if new_l != l || new_r != r {
                context.add(Expr::Div(new_l, new_r))
            } else {
                root
            }
        }
        Expr::Pow(b, e) => {
            let new_b = substitute_expr_by_id(context, b, target, replacement);
            let new_e = substitute_expr_by_id(context, e, target, replacement);
            if new_b != b || new_e != e {
                context.add(Expr::Pow(new_b, new_e))
            } else {
                root
            }
        }
        Expr::Neg(inner) => {
            let new_inner = substitute_expr_by_id(context, inner, target, replacement);
            if new_inner != inner {
                context.add(Expr::Neg(new_inner))
            } else {
                root
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args.iter() {
                let new_arg = substitute_expr_by_id(context, *arg, target, replacement);
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                context.add(Expr::Function(name, new_args))
            } else {
                root
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let mut new_data = Vec::new();
            let mut changed = false;
            for elem in data.iter() {
                let new_elem = substitute_expr_by_id(context, *elem, target, replacement);
                if new_elem != *elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                context.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                root
            }
        }
        Expr::Hold(inner) => {
            let new_inner = substitute_expr_by_id(context, inner, target, replacement);
            if new_inner != inner {
                context.add(Expr::Hold(new_inner))
            } else {
                root
            }
        }
        _ => root,
    }
}

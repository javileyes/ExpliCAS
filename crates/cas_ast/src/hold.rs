//! Hold Barrier Utilities
//!
//! `__hold(expr)` is an internal wrapper that blocks expansive/structural rules
//! but is transparent to basic algebra (AddView, MulView, cancellation).
//!
//! # Contract
//!
//! 1. `__hold` blocks: autoexpand, distribute, factor-undo rules
//! 2. `__hold` is transparent to: AddView, MulView, basic arithmetic
//! 3. `__hold` MUST be stripped before user-facing output (Display, JSON, FFI)
//!
//! # Usage
//!
//! - Rules that create protected results (expand, factor, poly_gcd) may wrap in __hold
//! - All output boundaries MUST call `strip_all_holds` before returning to user
//! - AddView/MulView SHOULD call `unwrap_hold` when collecting terms

use crate::{Context, Expr, ExprId};

/// Check if expression is wrapped in `__hold`
#[inline]
pub fn is_hold(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "__hold" && args.len() == 1)
}

/// Unwrap one level of `__hold` wrapper. Returns inner if hold, otherwise unchanged.
#[inline]
pub fn unwrap_hold(ctx: &Context, id: ExprId) -> ExprId {
    match ctx.get(id) {
        Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "__hold" && args.len() == 1 => {
            args[0]
        }
        _ => id,
    }
}

/// Unwrap `__hold` wrapper, returning `Some(inner)` if wrapped, `None` otherwise.
/// Useful for early-return patterns where you only want to act if it IS a hold wrapper.
#[inline]
pub fn unwrap_hold_if_wrapped(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "__hold" && args.len() == 1 => {
            Some(args[0])
        }
        _ => None,
    }
}

/// Recursively strip ALL `__hold()` wrappers from an expression tree.
/// Uses stack-based traversal to be safe for deep expressions.
///
/// This is the **canonical** implementation - do not duplicate elsewhere.
pub fn strip_all_holds(ctx: &mut Context, root: ExprId) -> ExprId {
    strip_holds_recursive(ctx, root)
}

/// Internal recursive implementation with memoization potential
fn strip_holds_recursive(ctx: &mut Context, id: ExprId) -> ExprId {
    match ctx.get(id).clone() {
        // Unwrap __hold and recurse into contents
        Expr::Function(fn_id, ref args) if ctx.sym_name(fn_id) == "__hold" && args.len() == 1 => {
            strip_holds_recursive(ctx, args[0])
        }

        // Binary operators - recurse into both sides
        Expr::Add(l, r) => {
            let new_l = strip_holds_recursive(ctx, l);
            let new_r = strip_holds_recursive(ctx, r);
            if new_l == l && new_r == r {
                id
            } else {
                ctx.add(Expr::Add(new_l, new_r))
            }
        }
        Expr::Sub(l, r) => {
            let new_l = strip_holds_recursive(ctx, l);
            let new_r = strip_holds_recursive(ctx, r);
            if new_l == l && new_r == r {
                id
            } else {
                ctx.add(Expr::Sub(new_l, new_r))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = strip_holds_recursive(ctx, l);
            let new_r = strip_holds_recursive(ctx, r);
            if new_l == l && new_r == r {
                id
            } else {
                ctx.add(Expr::Mul(new_l, new_r))
            }
        }
        Expr::Div(l, r) => {
            let new_l = strip_holds_recursive(ctx, l);
            let new_r = strip_holds_recursive(ctx, r);
            if new_l == l && new_r == r {
                id
            } else {
                ctx.add(Expr::Div(new_l, new_r))
            }
        }
        Expr::Pow(base, exp) => {
            let new_base = strip_holds_recursive(ctx, base);
            let new_exp = strip_holds_recursive(ctx, exp);
            if new_base == base && new_exp == exp {
                id
            } else {
                ctx.add(Expr::Pow(new_base, new_exp))
            }
        }

        // Unary
        Expr::Neg(inner) => {
            let new_inner = strip_holds_recursive(ctx, inner);
            if new_inner == inner {
                id
            } else {
                ctx.add(Expr::Neg(new_inner))
            }
        }

        // Functions (except __hold which is handled above)
        Expr::Function(fn_id, args) => {
            let mut changed = false;
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| {
                    let new_a = strip_holds_recursive(ctx, a);
                    if new_a != a {
                        changed = true;
                    }
                    new_a
                })
                .collect();
            if changed {
                ctx.add(Expr::Function(fn_id, new_args))
            } else {
                id
            }
        }

        // Matrix
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let new_data: Vec<ExprId> = data
                .iter()
                .map(|&e| {
                    let new_e = strip_holds_recursive(ctx, e);
                    if new_e != e {
                        changed = true;
                    }
                    new_e
                })
                .collect();
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                id
            }
        }

        // Leaves - no recursion needed
        Expr::Number(_) | Expr::Variable(_) | Expr::SessionRef(_) | Expr::Constant(_) => id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_hold() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let hold_x = ctx.call("__hold", vec![x]);

        assert!(!is_hold(&ctx, x));
        assert!(is_hold(&ctx, hold_x));
    }

    #[test]
    fn test_unwrap_hold() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let hold_x = ctx.call("__hold", vec![x]);

        assert_eq!(unwrap_hold(&ctx, x), x);
        assert_eq!(unwrap_hold(&ctx, hold_x), x);
    }

    #[test]
    fn test_strip_all_holds_nested() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        // __hold(__hold(x) + y)
        let hold_x = ctx.call("__hold", vec![x]);
        let sum = ctx.add(Expr::Add(hold_x, y));
        let outer_hold = ctx.call("__hold", vec![sum]);

        let result = strip_all_holds(&mut ctx, outer_hold);

        // Should be x + y (no holds)
        match ctx.get(result) {
            Expr::Add(l, r) => {
                assert_eq!(*l, x);
                assert_eq!(*r, y);
            }
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_strip_no_change_when_no_holds() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(x, y));

        let result = strip_all_holds(&mut ctx, sum);
        assert_eq!(result, sum, "Should return same ExprId when no holds");
    }
}

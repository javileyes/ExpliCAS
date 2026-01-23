// =============================================================================
// Semantic Trig Matchers (Ordered vs Unordered)
// =============================================================================
// These matchers enforce correct semantics for commutative (Add/Mul) vs
// non-commutative (Sub/Div) operations:
// - Add/Mul: arguments returned as UNORDERED set {A,B}
// - Sub/Div: arguments returned as ORDERED pair (A,B)
//
// RULE: The denominator can only VALIDATE args, never REORDER what was
// extracted from the numerator when the numerator is non-commutative.
// =============================================================================

use super::destructure::as_fn1;
use cas_ast::{Context, Expr, ExprId};

/// Result of matching a sum of trigonometric functions.
/// Arguments are UNORDERED (commutative).
#[derive(Debug, Clone, Copy)]
pub struct TrigSumMatch {
    /// First argument (order is canonical, not source-order)
    pub arg1: ExprId,
    /// Second argument
    pub arg2: ExprId,
}

/// Result of matching a difference of trigonometric functions.
/// Arguments are ORDERED (non-commutative, preserves source order).
#[derive(Debug, Clone, Copy)]
pub struct TrigDiffMatch {
    /// First argument (from minuend: sin(A) in sin(A)-sin(B))
    pub a: ExprId,
    /// Second argument (from subtrahend: sin(B) in sin(A)-sin(B))
    pub b: ExprId,
}

/// Match Add(trig(A), trig(B)) -> UNORDERED {A,B}
/// Returns arguments in canonical order for cache efficiency.
pub fn match_trig_sum(ctx: &Context, expr: ExprId, fn_name: &str) -> Option<TrigSumMatch> {
    if let Expr::Add(l, r) = ctx.get(expr) {
        let arg1 = as_fn1(ctx, *l, fn_name)?;
        let arg2 = as_fn1(ctx, *r, fn_name)?;
        // Note: order doesn't matter semantically, but we return as-is
        // (Context.add already canonicalizes Add order)
        return Some(TrigSumMatch { arg1, arg2 });
    }
    None
}

/// Match Sub(trig(A), trig(B)) -> ORDERED (A, B)
/// Also handles Add(trig(A), Neg(trig(B))) which is the canonicalized form.
///
/// CRITICAL: This preserves the ORDER from the source expression.
/// A comes from the minuend, B comes from the subtrahend.
/// This is essential for sign-correct computation of (A-B)/2.
pub fn match_trig_diff(ctx: &Context, expr: ExprId, fn_name: &str) -> Option<TrigDiffMatch> {
    // Pattern 1: Sub(trig(A), trig(B))
    if let Expr::Sub(l, r) = ctx.get(expr) {
        let a = as_fn1(ctx, *l, fn_name)?;
        let b = as_fn1(ctx, *r, fn_name)?;
        return Some(TrigDiffMatch { a, b });
    }

    // Pattern 2: Add(trig(A), Neg(trig(B))) - canonicalized form of subtraction
    if let Expr::Add(l, r) = ctx.get(expr) {
        // Check if r is Neg(trig(B))
        if let Expr::Neg(inner) = ctx.get(*r) {
            let a = as_fn1(ctx, *l, fn_name)?;
            let b = as_fn1(ctx, *inner, fn_name)?;
            return Some(TrigDiffMatch { a, b });
        }
        // Check if l is Neg(trig(A)) (less common but possible)
        if let Expr::Neg(inner) = ctx.get(*l) {
            // This is Neg(trig(A)) + trig(B) = trig(B) - trig(A)
            // So A is from r, B is from l's inner
            let a = as_fn1(ctx, *r, fn_name)?;
            let b = as_fn1(ctx, *inner, fn_name)?;
            // Note: this returns (B's arg, A's arg) as (a, b)
            // which represents trig(a) - trig(b)
            return Some(TrigDiffMatch { a, b });
        }
    }

    None
}

/// Check if two pairs of arguments match as unordered sets: {A,B} == {C,D}
///
/// This is used to validate that the denominator contains the same angles
/// as the numerator, WITHOUT reordering the numerator's extraction.
pub fn same_args_unordered(ctx: &Context, a: ExprId, b: ExprId, c: ExprId, d: ExprId) -> bool {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    // Canonicalize both pairs by putting smaller first
    let (a1, a2) = if compare_expr(ctx, a, b) == Ordering::Greater {
        (b, a)
    } else {
        (a, b)
    };
    let (b1, b2) = if compare_expr(ctx, c, d) == Ordering::Greater {
        (d, c)
    } else {
        (c, d)
    };

    // Compare canonicalized pairs
    compare_expr(ctx, a1, b1) == Ordering::Equal && compare_expr(ctx, a2, b2) == Ordering::Equal
}

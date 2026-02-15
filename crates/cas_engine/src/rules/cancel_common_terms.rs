//! Equation-level additive term cancellation utilities.
//!
//! Provides `cancel_common_additive_terms()`, which compares terms from
//! two expression trees (LHS and RHS of an equation) and cancels matching
//! pairs using structural comparison.
//!
//! # Why this is NOT a simplifier rule
//!
//! The simplifier's `CanonicalizeNegationRule` converts `Sub(a, b)` to
//! `Add(a, Neg(b))` very early in the pipeline. A rule targeting `Sub`
//! nodes would never fire because the Sub is already gone.
//!
//! More fundamentally, equation-level cancellation is a **relational
//! operation** between two expression trees (LHS vs RHS of `lhs = rhs`).
//! A simplifier rule only sees a single expression node. These are
//! different abstractions and belong in different layers.
//!
//! # Preconditions
//!
//! For maximum effectiveness, both `lhs` and `rhs` should be in
//! canonical form before calling:
//! - Rational exponents: `Div(Number(p), Number(q))` → `Number(p/q)`
//!   (via CanonicalizeRationalDivRule)
//! - Flattened Add/Mul chains with stable ordering
//!   (via standard canonicalization rules)
//!
//! # Guarantees
//!
//! - **Strictly reductive**: total term count always decreases
//! - **Sound**: uses `compare_expr` for structural equality (no numeric
//!   approximation)
//! - **Deterministic**: order of cancellation is fixed (rhs terms scanned
//!   in order, matched against lhs terms in order)

use crate::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

/// Result of equation-level additive term cancellation.
#[allow(dead_code)]
pub struct CancelResult {
    /// New LHS with cancelled terms removed.
    pub new_lhs: ExprId,
    /// New RHS with cancelled terms removed.
    pub new_rhs: ExprId,
    /// Number of term pairs that were cancelled.
    pub cancelled_count: usize,
}

/// Cancel common additive terms between two expression trees.
///
/// Flattens both sides into multisets of `(term, is_positive)`, matches
/// pairs by structural equality (same term, same sign polarity), and
/// rebuilds both sides without the matched pairs.
///
/// Returns `None` if no terms could be cancelled.
///
/// # Example
/// ```text
/// lhs = Add(x², Add(x^(5/6), 1))    rhs = x^(5/6)
/// → lhs_terms = [(x², +), (x^(5/6), +), (1, +)]
/// → rhs_terms = [(x^(5/6), +)]
/// → cancel x^(5/6) pair
/// → new_lhs = Add(x², 1), new_rhs = 0, cancelled = 1
/// ```
pub fn cancel_common_additive_terms(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelResult> {
    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_additive_terms(ctx, lhs, true, &mut lhs_terms);
    collect_additive_terms(ctx, rhs, true, &mut rhs_terms);

    let mut lhs_used = vec![false; lhs_terms.len()];
    let mut rhs_used = vec![false; rhs_terms.len()];
    let mut cancelled = 0;

    for (ri, (rt, rp)) in rhs_terms.iter().enumerate() {
        if rhs_used[ri] {
            continue;
        }
        for (li, (lt, lp)) in lhs_terms.iter().enumerate() {
            if lhs_used[li] {
                continue;
            }
            if lp == rp && compare_expr(ctx, *lt, *rt) == Ordering::Equal {
                lhs_used[li] = true;
                rhs_used[ri] = true;
                cancelled += 1;
                break;
            }
        }
    }

    if cancelled == 0 {
        return None;
    }

    let new_lhs_terms: Vec<_> = lhs_terms
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !lhs_used[*i])
        .map(|(_, t)| t)
        .collect();
    let new_rhs_terms: Vec<_> = rhs_terms
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !rhs_used[*i])
        .map(|(_, t)| t)
        .collect();

    Some(CancelResult {
        new_lhs: rebuild_from_terms(ctx, &new_lhs_terms),
        new_rhs: rebuild_from_terms(ctx, &new_rhs_terms),
        cancelled_count: cancelled,
    })
}

/// Collect additive terms from an expression, flattening Add/Sub/Neg.
/// Each term is `(ExprId, is_positive)`.
pub(crate) fn collect_additive_terms(
    ctx: &Context,
    id: ExprId,
    positive: bool,
    out: &mut Vec<(ExprId, bool)>,
) {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            collect_additive_terms(ctx, *l, positive, out);
            collect_additive_terms(ctx, *r, positive, out);
        }
        Expr::Sub(l, r) => {
            collect_additive_terms(ctx, *l, positive, out);
            collect_additive_terms(ctx, *r, !positive, out);
        }
        Expr::Neg(inner) => {
            collect_additive_terms(ctx, *inner, !positive, out);
        }
        _ => {
            out.push((id, positive));
        }
    }
}

/// Rebuild an expression from a list of `(term, is_positive)` pairs.
/// Returns `Number(0)` for an empty list.
pub(crate) fn rebuild_from_terms(ctx: &mut Context, terms: &[(ExprId, bool)]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }

    let mut result = if terms[0].1 {
        terms[0].0
    } else {
        ctx.add(Expr::Neg(terms[0].0))
    };

    for &(term, positive) in &terms[1..] {
        if positive {
            result = ctx.add(Expr::Add(result, term));
        } else {
            result = ctx.add(Expr::Sub(result, term));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr};

    #[test]
    fn test_cancel_simple() {
        // (x^2 + y) - y → x^2 (on LHS), 0 (on RHS)
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x2 = ctx.add(Expr::Pow(x, two));
        let y = ctx.var("y");
        let lhs = ctx.add(Expr::Add(x2, y));
        let rhs = y;
        let result = cancel_common_additive_terms(&mut ctx, lhs, rhs).unwrap();
        assert_eq!(result.cancelled_count, 1);
        assert!(matches!(ctx.get(result.new_lhs), Expr::Pow(_, _)));
        assert!(matches!(ctx.get(result.new_rhs), Expr::Number(_))); // 0
    }

    #[test]
    fn test_no_cancel_different_terms() {
        // (x + y) vs z → no cancellation
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let lhs = ctx.add(Expr::Add(x, y));
        assert!(cancel_common_additive_terms(&mut ctx, lhs, z).is_none());
    }

    #[test]
    fn test_cancel_with_duplicates() {
        // (a + b + b) vs b → cancels one b, leaves (a + b) vs 0
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let b2 = ctx.var("b");
        let ab = ctx.add(Expr::Add(a, b));
        let lhs = ctx.add(Expr::Add(ab, b2));
        let result = cancel_common_additive_terms(&mut ctx, lhs, b).unwrap();
        assert_eq!(result.cancelled_count, 1);
    }

    #[test]
    fn test_cancel_symmetric() {
        // (a + b + c) vs (b + c) → a vs 0, cancelled=2
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let ab = ctx.add(Expr::Add(a, b));
        let lhs = ctx.add(Expr::Add(ab, c));
        let rhs = ctx.add(Expr::Add(b, c));
        let result = cancel_common_additive_terms(&mut ctx, lhs, rhs).unwrap();
        assert_eq!(result.cancelled_count, 2);
        // new_lhs should be a, new_rhs should be 0
        assert!(matches!(ctx.get(result.new_lhs), Expr::Variable(_)));
        assert!(matches!(ctx.get(result.new_rhs), Expr::Number(_)));
    }
}

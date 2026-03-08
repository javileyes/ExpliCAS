//! Shared helpers for equation-level additive cancellation flows.
//!
//! These utilities are pure AST operations and are intentionally kept in
//! `cas_math` so engine-layer modules can stay focused on orchestration.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// Collect additive terms from an expression, flattening `Add/Sub/Neg`.
///
/// Each term is returned as `(term_expr_id, is_positive)`.
pub fn collect_additive_terms_signed(
    ctx: &Context,
    id: ExprId,
    positive: bool,
    out: &mut Vec<(ExprId, bool)>,
) {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            collect_additive_terms_signed(ctx, *l, positive, out);
            collect_additive_terms_signed(ctx, *r, positive, out);
        }
        Expr::Sub(l, r) => {
            collect_additive_terms_signed(ctx, *l, positive, out);
            collect_additive_terms_signed(ctx, *r, !positive, out);
        }
        Expr::Neg(inner) => {
            collect_additive_terms_signed(ctx, *inner, !positive, out);
        }
        _ => out.push((id, positive)),
    }
}

/// Rebuild an additive expression from signed terms.
///
/// Empty input returns `0`.
pub fn rebuild_from_signed_terms(ctx: &mut Context, terms: &[(ExprId, bool)]) -> ExprId {
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

/// Compute a deterministic structural hash of an AST subtree.
///
/// This hash is shape-sensitive and order-sensitive for non-commutative nodes.
/// It is suitable as a cheap pre-filter before exact structural comparison.
pub fn structural_expr_fingerprint(ctx: &Context, id: ExprId) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    hash_expr_structural(ctx, id, &mut h);
    h.finish()
}

fn hash_expr_structural(ctx: &Context, id: ExprId, h: &mut impl Hasher) {
    let node = ctx.get(id);
    std::mem::discriminant(node).hash(h);
    match node {
        Expr::Number(n) => n.hash(h),
        Expr::Variable(s) => ctx.sym_name(*s).hash(h),
        Expr::Constant(c) => std::mem::discriminant(c).hash(h),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            hash_expr_structural(ctx, *l, h);
            hash_expr_structural(ctx, *r, h);
        }
        Expr::Pow(b, e) => {
            hash_expr_structural(ctx, *b, h);
            hash_expr_structural(ctx, *e, h);
        }
        Expr::Neg(e) | Expr::Hold(e) => hash_expr_structural(ctx, *e, h),
        Expr::Function(name, args) => {
            ctx.sym_name(*name).hash(h);
            for a in args {
                hash_expr_structural(ctx, *a, h);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            rows.hash(h);
            cols.hash(h);
            for d in data {
                hash_expr_structural(ctx, *d, h);
            }
        }
        Expr::SessionRef(s) => s.hash(h),
    }
}

/// Lightweight 2-term product simplification for preview fingerprinting.
///
/// Folds `Pow(x,a) * Pow(x,b) -> Pow(x,a+b)` and `Number * Number -> Number`,
/// treating bare variables as `Pow(x, 1)`. Falls back to plain multiplication.
pub fn mul_preview(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if let (Expr::Number(na), Expr::Number(nb)) = (ctx.get(a), ctx.get(b)) {
        let product = na.clone() * nb.clone();
        return ctx.add(Expr::Number(product));
    }

    if matches!(ctx.get(a), Expr::Number(_)) || matches!(ctx.get(b), Expr::Number(_)) {
        return ctx.add(Expr::Mul(a, b));
    }

    let (ba, ea) = match ctx.get(a) {
        Expr::Pow(base, exp) => (*base, Some(*exp)),
        _ => (a, None),
    };
    let (bb, eb) = match ctx.get(b) {
        Expr::Pow(base, exp) => (*base, Some(*exp)),
        _ => (b, None),
    };

    if compare_expr(ctx, ba, bb) == Ordering::Equal {
        let ea_val = ea.and_then(|e| {
            if let Expr::Number(n) = ctx.get(e) {
                Some(n.clone())
            } else {
                None
            }
        });
        let eb_val = eb.and_then(|e| {
            if let Expr::Number(n) = ctx.get(e) {
                Some(n.clone())
            } else {
                None
            }
        });
        let ea_num = ea_val.unwrap_or_else(|| BigRational::from_integer(1.into()));
        let eb_num = eb_val.unwrap_or_else(|| BigRational::from_integer(1.into()));
        let sum = ea_num + eb_num;
        let new_exp = ctx.add(Expr::Number(sum));
        return ctx.add(Expr::Pow(ba, new_exp));
    }

    ctx.add(Expr::Mul(a, b))
}

/// Result for structural cancellation between two additive expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelCommonAdditivePlan {
    pub new_lhs: ExprId,
    pub new_rhs: ExprId,
    pub cancelled_count: usize,
}

/// Cancel common additive terms between two expressions using structural
/// equality and sign-aware matching.
///
/// This performs a pure AST transformation:
/// - flatten both sides as signed additive terms,
/// - match equal-sign/equal-structure pairs,
/// - rebuild each side with unmatched terms.
pub fn try_cancel_common_additive_terms_expr(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelCommonAdditivePlan> {
    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_additive_terms_signed(ctx, lhs, true, &mut lhs_terms);
    collect_additive_terms_signed(ctx, rhs, true, &mut rhs_terms);

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

    Some(CancelCommonAdditivePlan {
        new_lhs: rebuild_from_signed_terms(ctx, &new_lhs_terms),
        new_rhs: rebuild_from_signed_terms(ctx, &new_rhs_terms),
        cancelled_count: cancelled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_compare::poly_eq;
    use cas_ast::ordering::compare_expr;
    use cas_parser::parse;

    #[test]
    fn collect_and_rebuild_signed_terms_roundtrip() {
        let mut ctx = Context::new();
        let expr = parse("a - b + c", &mut ctx).expect("parse");
        let mut terms = Vec::new();
        collect_additive_terms_signed(&ctx, expr, true, &mut terms);
        let rebuilt = rebuild_from_signed_terms(&mut ctx, &terms);
        assert!(poly_eq(&ctx, expr, rebuilt));
    }

    #[test]
    fn fingerprint_is_stable_for_same_expr() {
        let mut ctx = Context::new();
        let expr = parse("(x + 1)^2", &mut ctx).expect("parse");
        let h1 = structural_expr_fingerprint(&ctx, expr);
        let h2 = structural_expr_fingerprint(&ctx, expr);
        assert_eq!(h1, h2);
    }

    #[test]
    fn mul_preview_merges_equal_power_bases() {
        let mut ctx = Context::new();
        let expr = parse("x^2 * x^3", &mut ctx).expect("parse");
        let (a, b) = match ctx.get(expr) {
            Expr::Mul(l, r) => (*l, *r),
            _ => panic!("expected mul"),
        };
        let merged = mul_preview(&mut ctx, a, b);
        let expected = parse("x^5", &mut ctx).expect("parse");
        assert_eq!(compare_expr(&ctx, merged, expected), Ordering::Equal);
    }

    #[test]
    fn structural_cancel_common_additive_terms_basic() {
        let mut ctx = Context::new();
        let lhs = parse("x^2 + y", &mut ctx).expect("parse");
        let rhs = parse("y", &mut ctx).expect("parse");
        let plan = try_cancel_common_additive_terms_expr(&mut ctx, lhs, rhs).expect("plan");
        assert_eq!(plan.cancelled_count, 1);
        let expected_lhs = parse("x^2", &mut ctx).expect("parse");
        let expected_rhs = parse("0", &mut ctx).expect("parse");
        assert_eq!(
            compare_expr(&ctx, plan.new_lhs, expected_lhs),
            Ordering::Equal
        );
        assert_eq!(
            compare_expr(&ctx, plan.new_rhs, expected_rhs),
            Ordering::Equal
        );
    }
}

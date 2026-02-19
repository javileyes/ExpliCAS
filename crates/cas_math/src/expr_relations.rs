//! Structural relation helpers between expressions.
//!
//! These predicates are used by polynomial/factoring rules to detect
//! negation and conjugate relationships without binding to engine modules.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;
use std::cmp::Ordering;

/// Check if two expressions are structural negations of each other.
///
/// Supports:
/// - `Neg(a)`
/// - `Mul(-1, a)` and `Mul(a, -1)`
pub fn is_negation(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    check_negation_structure(ctx, b, a) || check_negation_structure(ctx, a, b)
}

fn check_negation_structure(ctx: &Context, potential_neg: ExprId, original: ExprId) -> bool {
    match ctx.get(potential_neg) {
        Expr::Neg(n) => compare_expr(ctx, original, *n) == Ordering::Equal,
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == -BigRational::one() && compare_expr(ctx, *r, original) == Ordering::Equal {
                    return true;
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == -BigRational::one() && compare_expr(ctx, *l, original) == Ordering::Equal {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

/// Check whether `a` and `b` are a conjugate additive pair.
///
/// Recognizes:
/// - `(A + B)` with `(A - B)` (order-insensitive on additive terms)
/// - canonicalized additive variants like `(A + B)` with `(A + (-B))`
pub fn is_conjugate_add_sub(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let a_expr = ctx.get(a);
    let b_expr = ctx.get(b);

    match (a_expr, b_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            if compare_expr(ctx, a1, b1) == Ordering::Equal
                && compare_expr(ctx, a2, b2) == Ordering::Equal
            {
                return true;
            }
            if compare_expr(ctx, a2, b1) == Ordering::Equal
                && compare_expr(ctx, a1, b2) == Ordering::Equal
            {
                return true;
            }
            false
        }
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            if is_negation(ctx, a2, b2) && compare_expr(ctx, a1, b1) == Ordering::Equal {
                return true;
            }
            if is_negation(ctx, a2, b1) && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return true;
            }
            if is_negation(ctx, a1, b2) && compare_expr(ctx, a2, b1) == Ordering::Equal {
                return true;
            }
            if is_negation(ctx, a1, b1) && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return true;
            }
            false
        }
        _ => false,
    }
}

/// Count additive terms by flattening `Add/Sub` recursively.
///
/// `Neg` preserves term count of its inner expression.
pub fn count_additive_terms(ctx: &Context, expr: ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Add(l, r) => count_additive_terms(ctx, *l) + count_additive_terms(ctx, *r),
        Expr::Sub(l, r) => count_additive_terms(ctx, *l) + count_additive_terms(ctx, *r),
        Expr::Neg(inner) => count_additive_terms(ctx, *inner),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn negation_detection_covers_neg_and_mul_minus_one() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let neg_x = parse("-x", &mut ctx).expect("-x");
        let mul_neg_x = parse("(-1)*x", &mut ctx).expect("(-1)*x");
        assert!(is_negation(&ctx, x, neg_x));
        assert!(is_negation(&ctx, x, mul_neg_x));
    }

    #[test]
    fn conjugate_detection_for_add_sub_and_add_neg() {
        let mut ctx = Context::new();
        let add = parse("a+b", &mut ctx).expect("a+b");
        let sub = parse("a-b", &mut ctx).expect("a-b");
        let add_neg = parse("a+(-b)", &mut ctx).expect("a+(-b)");
        let same = parse("a+b", &mut ctx).expect("a+b");
        assert!(is_conjugate_add_sub(&ctx, add, sub));
        assert!(is_conjugate_add_sub(&ctx, add, add_neg));
        assert!(!is_conjugate_add_sub(&ctx, add, same));
    }

    #[test]
    fn additive_term_count_handles_nested_sub_and_neg() {
        let mut ctx = Context::new();
        let nested = parse("a-(b-c)", &mut ctx).expect("nested");
        let neg = parse("-(a+b)", &mut ctx).expect("neg");
        assert_eq!(count_additive_terms(&ctx, nested), 3);
        assert_eq!(count_additive_terms(&ctx, neg), 2);
    }
}

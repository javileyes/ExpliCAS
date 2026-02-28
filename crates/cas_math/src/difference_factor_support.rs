//! Helpers for detecting and constructing difference factors.

use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Signed;

/// Detect whether `expr` encodes a difference `x - y`.
///
/// Supports canonical and near-canonical forms:
/// - `Sub(x, y)`
/// - `Add(x, Neg(y))`
/// - `Add(x, Mul(-1, y))`
/// - `Add(Neg(x), y)` (interpreted as `y - x`)
pub fn extract_difference_pair(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(l, r) => Some((*l, *r)),
        Expr::Add(l, r) => {
            if let Expr::Neg(inner) = ctx.get(*r) {
                return Some((*l, *inner));
            }
            if let Expr::Mul(a, b) = ctx.get(*r) {
                if let Expr::Number(n) = ctx.get(*a) {
                    if n.is_negative() && *n == BigRational::from_integer((-1).into()) {
                        return Some((*l, *b));
                    }
                }
                if let Expr::Number(n) = ctx.get(*b) {
                    if n.is_negative() && *n == BigRational::from_integer((-1).into()) {
                        return Some((*l, *a));
                    }
                }
            }
            if let Expr::Neg(inner) = ctx.get(*l) {
                return Some((*r, *inner));
            }
            None
        }
        _ => None,
    }
}

/// Build a difference expression `x - y`.
pub fn build_difference_expr(ctx: &mut Context, x: ExprId, y: ExprId) -> ExprId {
    ctx.add(Expr::Sub(x, y))
}

#[cfg(test)]
mod tests {
    use super::{build_difference_expr, extract_difference_pair};
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn extracts_add_neg_form() {
        let mut ctx = Context::new();
        let expr = parse("a + (-b)", &mut ctx).expect("parse");
        let (x, y) = extract_difference_pair(&ctx, expr).expect("diff pair");
        let expected_x = parse("a", &mut ctx).expect("parse x");
        let expected_y = parse("b", &mut ctx).expect("parse y");
        assert_eq!(compare_expr(&ctx, x, expected_x), Ordering::Equal);
        assert_eq!(compare_expr(&ctx, y, expected_y), Ordering::Equal);
    }

    #[test]
    fn build_difference_returns_sub_node() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let y = parse("y", &mut ctx).expect("parse");
        let d = build_difference_expr(&mut ctx, x, y);
        assert!(matches!(ctx.get(d), cas_ast::Expr::Sub(_, _)));
    }
}

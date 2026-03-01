//! Support for matching `a - b/a`-like fraction subtraction patterns.

use crate::build::mul2_raw;
use crate::expr_destructure::{as_add, as_sub};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct SubTermMatchesDenomRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Try to rewrite `a - b/a` (canonicalized as `a + -(b/a)`) to `(a^2 - b)/a`.
///
/// Also handles the commuted add form `-(b/a) + a`.
pub fn try_rewrite_sub_term_matches_denom_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SubTermMatchesDenomRewrite> {
    let (term, p, q) = if let Some((l, r)) = as_sub(ctx, expr) {
        if let Expr::Div(p, q) = ctx.get(r) {
            (l, *p, *q)
        } else {
            return None;
        }
    } else {
        let (l, r) = as_add(ctx, expr)?;
        if let Expr::Neg(inner) = ctx.get(r) {
            if let Expr::Div(p, q) = ctx.get(*inner) {
                (l, *p, *q)
            } else {
                return None;
            }
        } else if let Expr::Neg(inner) = ctx.get(l) {
            if let Expr::Div(p, q) = ctx.get(*inner) {
                (r, *p, *q)
            } else {
                return None;
            }
        } else {
            return None;
        }
    };

    if compare_expr(ctx, q, term) != std::cmp::Ordering::Equal {
        return None;
    }

    let term_squared = mul2_raw(ctx, term, term);
    let new_num = ctx.add(Expr::Sub(term_squared, p));
    let new_expr = ctx.add(Expr::Div(new_num, term));

    Some(SubTermMatchesDenomRewrite {
        rewritten: new_expr,
        desc: "Common denominator: a - b/a → (a² - b)/a",
    })
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_sub_term_matches_denom_expr;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_ast::Expr;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn rewrites_basic_sub_term_matches_denom() {
        let mut ctx = Context::new();
        let expr = parse("a - b/a", &mut ctx).expect("parse");
        let rw = try_rewrite_sub_term_matches_denom_expr(&mut ctx, expr).expect("rewrite");
        let a = parse("a", &mut ctx).expect("parse");
        if let Expr::Div(_, den) = ctx.get(rw.rewritten) {
            assert_eq!(compare_expr(&ctx, *den, a), Ordering::Equal);
        } else {
            panic!("expected division rewrite");
        }
    }

    #[test]
    fn rewrites_commuted_canonical_form() {
        let mut ctx = Context::new();
        let expr = parse("-(b/a) + a", &mut ctx).expect("parse");
        let rw = try_rewrite_sub_term_matches_denom_expr(&mut ctx, expr).expect("rewrite");
        let a = parse("a", &mut ctx).expect("parse");
        if let Expr::Div(_, den) = ctx.get(rw.rewritten) {
            assert_eq!(compare_expr(&ctx, *den, a), Ordering::Equal);
        } else {
            panic!("expected division rewrite");
        }
    }

    #[test]
    fn does_not_rewrite_when_denominator_differs() {
        let mut ctx = Context::new();
        let expr = parse("a - b/c", &mut ctx).expect("parse");
        assert!(try_rewrite_sub_term_matches_denom_expr(&mut ctx, expr).is_none());
    }
}

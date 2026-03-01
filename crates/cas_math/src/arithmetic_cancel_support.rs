//! Pattern helpers for arithmetic self-cancellation rewrites.

use crate::semantic_equality::SemanticEqualityChecker;
use cas_ast::{Context, Expr, ExprId};

/// Match `a - a` using semantic equality.
///
/// Returns the representative inner term when both sides are semantically equal.
pub fn match_sub_self_semantic_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
        return None;
    };

    let checker = SemanticEqualityChecker::new(ctx);
    if checker.are_equal(*lhs, *rhs) {
        Some(*lhs)
    } else {
        None
    }
}

/// Match additive inverse patterns:
/// - `a + (-a)`
/// - `(-a) + a`
///
/// Returns `a` when matched.
pub fn match_add_inverse_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };

    if let Expr::Neg(neg_inner) = ctx.get(*r) {
        if *neg_inner == *l {
            return Some(*l);
        }
    }
    if let Expr::Neg(neg_inner) = ctx.get(*l) {
        if *neg_inner == *r {
            return Some(*r);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{match_add_inverse_expr, match_sub_self_semantic_expr};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn detects_sub_self_symbolic() {
        let mut ctx = Context::new();
        let expr = parse("tan(3*x)-tan(3*x)", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_add_inverse_both_orders() {
        let mut ctx = Context::new();
        let expr1 = parse("a+(-a)", &mut ctx).expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr1).is_some());

        let expr2 = parse("(-a)+a", &mut ctx).expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr2).is_some());
    }
}

//! Pattern helpers for arithmetic self-cancellation rewrites.

use crate::semantic_equality::SemanticEqualityChecker;
use cas_ast::{Context, Expr, ExprId};

fn extract_abs_sub_like_pair(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs) || args.len() != 1 {
        return None;
    }

    let mut scratch = ctx.clone();
    crate::expr_sub_like::extract_sub_like_pair(&mut scratch, args[0])
}

fn match_abs_sub_mirror_expr(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    let Some((l1, l2)) = extract_abs_sub_like_pair(ctx, lhs) else {
        return false;
    };
    let Some((r1, r2)) = extract_abs_sub_like_pair(ctx, rhs) else {
        return false;
    };

    let checker = SemanticEqualityChecker::new(ctx);
    checker.are_equal(l1, r2) && checker.are_equal(l2, r1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArithmeticCancelRewrite {
    pub rewritten: ExprId,
    pub inner: ExprId,
}

/// Match `a - a` using semantic equality.
///
/// Returns the representative inner term when both sides are semantically equal.
pub fn match_sub_self_semantic_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
        return None;
    };

    if match_abs_sub_mirror_expr(ctx, *lhs, *rhs) {
        return Some(*lhs);
    }

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

    let checker = SemanticEqualityChecker::new(ctx);

    if let Expr::Neg(neg_inner) = ctx.get(*r) {
        if match_abs_sub_mirror_expr(ctx, *neg_inner, *l) {
            return Some(*l);
        }
        if *neg_inner == *l || checker.are_equal(*neg_inner, *l) {
            return Some(*l);
        }
    }
    if let Expr::Neg(neg_inner) = ctx.get(*l) {
        if match_abs_sub_mirror_expr(ctx, *neg_inner, *r) {
            return Some(*r);
        }
        if *neg_inner == *r || checker.are_equal(*neg_inner, *r) {
            return Some(*r);
        }
    }

    None
}

/// Rewrite `a - a` to `0`, returning the cancelled inner expression.
pub fn try_rewrite_sub_self_zero_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticCancelRewrite> {
    let inner = match_sub_self_semantic_expr(ctx, expr)?;
    Some(ArithmeticCancelRewrite {
        rewritten: ctx.num(0),
        inner,
    })
}

/// Rewrite `a + (-a)` (or `(-a) + a`) to `0`, returning `a`.
pub fn try_rewrite_add_inverse_zero_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticCancelRewrite> {
    let inner = match_add_inverse_expr(ctx, expr)?;
    Some(ArithmeticCancelRewrite {
        rewritten: ctx.num(0),
        inner,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        match_add_inverse_expr, match_sub_self_semantic_expr, try_rewrite_add_inverse_zero_expr,
        try_rewrite_sub_self_zero_expr,
    };
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn detects_sub_self_symbolic() {
        let mut ctx = Context::new();
        let expr = parse("tan(3*x)-tan(3*x)", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_sub_self_with_abs_sub_mirror_forms() {
        let mut ctx = Context::new();
        let expr = parse(
            "abs((1/(x - 1) + 1/(x + 1)) - 1) - abs(1 - (1/(x - 1) + 1/(x + 1)))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_sub_self_with_abs_sub_mirror_runtime_shape() {
        let mut ctx = Context::new();
        let expr = parse(
            "abs((2*x)/(x^2 - 1) - 1) - abs(1 - 2*x/(x^2 - 1))",
            &mut ctx,
        )
        .expect("parse");
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

    #[test]
    fn detects_add_inverse_with_abs_sub_mirror_forms() {
        let mut ctx = Context::new();
        let expr = parse(
            "abs((2*x)/(x^2 - 1) - 1) + (-abs(1 - 2*x/(x^2 - 1)))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_add_inverse_with_semantically_equal_trig_wrappers() {
        let mut ctx = Context::new();
        let expr = parse(
            "sin((2*u + 1)/(u*(u+1))) + (-sin((2*u + 1)/(u^2 + u)))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn rewrites_sub_self_to_zero() {
        let mut ctx = Context::new();
        let expr = parse("x-x", &mut ctx).expect("parse");
        let rewrite = try_rewrite_sub_self_zero_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "0"
        );
    }

    #[test]
    fn rewrites_add_inverse_to_zero() {
        let mut ctx = Context::new();
        let expr = parse("x+(-x)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_add_inverse_zero_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "0"
        );
    }
}

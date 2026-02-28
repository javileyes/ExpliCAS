//! Structural helpers for simple multiplicative distribution rewrites.

use crate::expr_rewrite::distribute;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleDistributionRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Rewrite one-step multiplicative distribution:
/// - `a*(b+c)` -> `a*b + a*c`
/// - `(b+c)*a` -> `b*a + c*a`
/// - same for subtraction.
pub fn try_rewrite_simple_mul_distribution_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SimpleDistributionRewrite> {
    let (left, right) = match ctx.get(expr) {
        Expr::Mul(l, r) => (*l, *r),
        _ => return None,
    };

    if matches!(ctx.get(right), Expr::Add(_, _) | Expr::Sub(_, _)) {
        let rewritten = distribute(ctx, right, left);
        if rewritten != expr {
            return Some(SimpleDistributionRewrite {
                rewritten,
                desc: "Distribute (RHS)",
            });
        }
    }

    if matches!(ctx.get(left), Expr::Add(_, _) | Expr::Sub(_, _)) {
        let rewritten = distribute(ctx, left, right);
        if rewritten != expr {
            return Some(SimpleDistributionRewrite {
                rewritten,
                desc: "Distribute (LHS)",
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_simple_mul_distribution_expr;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn rewrites_rhs_additive_product() {
        let mut ctx = Context::new();
        let expr = parse("a*(b+c)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_simple_mul_distribution_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "Distribute (RHS)");
        assert!(matches!(ctx.get(rewrite.rewritten), Expr::Add(_, _)));
    }

    #[test]
    fn rewrites_lhs_additive_product() {
        let mut ctx = Context::new();
        let expr = parse("(b-c)*a", &mut ctx).expect("parse");
        let rewrite = try_rewrite_simple_mul_distribution_expr(&mut ctx, expr).expect("rewrite");
        assert!(rewrite.desc == "Distribute (LHS)" || rewrite.desc == "Distribute (RHS)");
        assert!(matches!(
            ctx.get(rewrite.rewritten),
            Expr::Add(_, _) | Expr::Sub(_, _)
        ));
    }
}

//! Pre-order planning helpers for power rewrites.

use crate::expr_predicates::is_half_expr;
use crate::sqrt_square_support::{detect_sqrt_square_pattern, SqrtSquarePattern};
use cas_ast::{Context, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqrtSquarePowRewritePlan {
    pub rewritten: ExprId,
    pub identity_desc: &'static str,
    pub rule_name: &'static str,
}

/// Plan `(u^2)^(1/2)` and `(u*u)^(1/2)` rewrites as `abs(u)`.
pub fn try_plan_sqrt_square_pow_rewrite(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
) -> Option<SqrtSquarePowRewritePlan> {
    if !is_half_expr(ctx, exp) {
        return None;
    }

    let (arg, identity_desc, rule_name) = match detect_sqrt_square_pattern(ctx, base)? {
        SqrtSquarePattern::PowSquare { arg } => {
            (arg, "sqrt(u^2) = |u|", "Simplify Square Root of Square")
        }
        SqrtSquarePattern::RepeatedMul { arg } => {
            (arg, "sqrt(u * u) = |u|", "Simplify Square Root of Product")
        }
    };

    let rewritten = ctx.call("abs", vec![arg]);
    Some(SqrtSquarePowRewritePlan {
        rewritten,
        identity_desc,
        rule_name,
    })
}

#[cfg(test)]
mod tests {
    use super::try_plan_sqrt_square_pow_rewrite;
    use cas_ast::{Context, Expr};

    #[test]
    fn plans_pow_square_half_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let half = ctx.rational(1, 2);
        let base = ctx.add(Expr::Pow(x, two));
        let exp = half;
        let plan = try_plan_sqrt_square_pow_rewrite(&mut ctx, base, exp).expect("plan");
        assert_eq!(cas_formatter::render_expr(&ctx, plan.rewritten), "|x|");
    }

    #[test]
    fn plans_repeated_mul_half_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let half = ctx.rational(1, 2);
        let base = ctx.add(Expr::Mul(x, x));
        let exp = half;
        let plan = try_plan_sqrt_square_pow_rewrite(&mut ctx, base, exp).expect("plan");
        assert_eq!(cas_formatter::render_expr(&ctx, plan.rewritten), "|x|");
    }

    #[test]
    fn rejects_non_half_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let base = ctx.add(Expr::Pow(x, two));
        let exp = three;
        assert!(try_plan_sqrt_square_pow_rewrite(&mut ctx, base, exp).is_none());
    }
}

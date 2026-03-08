//! Support helpers for rational canonicalization rewrites.

use crate::expr_destructure::{as_div, as_pow};
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::Zero;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalCanonicalizationRewrite {
    pub rewritten: ExprId,
}

/// Rewrite `Div(Number(p), Number(q)) -> Number(p/q)` when denominator is non-zero.
pub fn try_rewrite_rational_div_canonical_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RationalCanonicalizationRewrite> {
    let (num, den) = as_div(ctx, expr)?;

    let (p, q) = match (ctx.get(num), ctx.get(den)) {
        (Expr::Number(p), Expr::Number(q)) => (p.clone(), q.clone()),
        _ => return None,
    };

    if q.is_zero() {
        return None;
    }

    let rewritten = ctx.add(Expr::Number(&p / &q));
    Some(RationalCanonicalizationRewrite { rewritten })
}

/// Rewrite `Pow(Pow(base, k), r) -> Pow(base, k*r)` when domain-safe in reals.
///
/// Safety condition: skip only the unsafe case where both `k` and denominator(`r`) are even.
pub fn try_rewrite_nested_pow_canonical_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RationalCanonicalizationRewrite> {
    let (outer_base, outer_exp) = as_pow(ctx, expr)?;
    let (inner_base, inner_exp) = as_pow(ctx, outer_base)?;

    let k = match ctx.get(inner_exp) {
        Expr::Number(n) => n.clone(),
        _ => return None,
    };
    let r = match ctx.get(outer_exp) {
        Expr::Number(n) => n.clone(),
        _ => return None,
    };

    let q_is_even = r.denom().is_even();
    if q_is_even {
        if !k.is_integer() {
            return None;
        }
        let k_int = k.to_integer();
        if k_int.is_even() {
            return None;
        }
    }

    let new_exp = ctx.add(Expr::Number(&k * &r));
    let rewritten = ctx.add(Expr::Pow(inner_base, new_exp));
    Some(RationalCanonicalizationRewrite { rewritten })
}

#[cfg(test)]
mod tests {
    use super::{try_rewrite_nested_pow_canonical_expr, try_rewrite_rational_div_canonical_expr};
    use cas_ast::{Context, Expr};

    #[test]
    fn rewrites_rational_division_into_single_number() {
        let mut ctx = Context::new();
        let five = ctx.num(5);
        let six = ctx.num(6);
        let expr = ctx.add(Expr::Div(five, six));
        let rewrite = try_rewrite_rational_div_canonical_expr(&mut ctx, expr).expect("rewrite");
        match ctx.get(rewrite.rewritten) {
            Expr::Number(n) => assert_eq!(n.to_string(), "5/6"),
            got => panic!("expected number, got {:?}", got),
        }
    }

    #[test]
    fn blocks_rational_division_when_denominator_is_zero() {
        let mut ctx = Context::new();
        let five = ctx.num(5);
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Div(five, zero));
        assert!(try_rewrite_rational_div_canonical_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn rewrites_safe_nested_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three = ctx.num(3);
        let inner = ctx.add(Expr::Pow(x, three));
        let half = ctx.rational(1, 2);
        let expr = ctx.add(Expr::Pow(inner, half));
        let rewrite = try_rewrite_nested_pow_canonical_expr(&mut ctx, expr).expect("rewrite");
        match ctx.get(rewrite.rewritten) {
            Expr::Pow(_, exp) => match ctx.get(*exp) {
                Expr::Number(n) => assert_eq!(n.to_string(), "3/2"),
                got => panic!("expected number exponent, got {:?}", got),
            },
            got => panic!("expected pow, got {:?}", got),
        }
    }

    #[test]
    fn blocks_unsafe_nested_power_even_k_even_root() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let inner = ctx.add(Expr::Pow(x, two));
        let half = ctx.rational(1, 2);
        let expr = ctx.add(Expr::Pow(inner, half));
        assert!(try_rewrite_nested_pow_canonical_expr(&mut ctx, expr).is_none());
    }
}

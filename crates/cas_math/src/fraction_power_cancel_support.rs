//! Support for power/fraction cancellation rewrites.
//!
//! This module contains structural/algebraic rewrite planning for:
//! - `P^m / P^n -> P^(m-n)` (integer exponents)
//! - `P / P -> 1`
//! - `P^n / P -> P^(n-1)` and `P^n / (-P) -> -P^(n-1)`
//!
//! Domain-mode policy (strict/assume) is intentionally left to callers.

use crate::expr_destructure::{as_div, as_pow};
use crate::numeric::as_i64;
use crate::poly_compare::{poly_relation, SignRelation};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone)]
pub struct CancelSameBasePowersRewrite {
    pub rewritten: ExprId,
    pub nonzero_target: ExprId,
    pub kind: CancelSameBasePowersRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelSameBasePowersRewriteKind {
    EqualPowers,
    CollapseToBase,
    CollapseToReciprocalBase,
    CollapseToPositivePower(i64),
    CollapseToReciprocalPower(i64),
}

#[derive(Debug, Clone)]
pub struct CancelIdenticalFractionRewrite {
    pub rewritten: ExprId,
    pub nonzero_target: ExprId,
    pub kind: CancelIdenticalFractionRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelIdenticalFractionRewriteKind {
    IdenticalFraction,
}

#[derive(Debug, Clone)]
pub struct CancelPowerFractionRewrite {
    pub rewritten: ExprId,
    pub nonzero_target: ExprId,
    pub kind: CancelPowerFractionRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelPowerFractionRewriteKind {
    SameSign,
    NegatedDenominator,
}

/// Try to rewrite `P^m / P^n` into a simpler power form (integer exponents).
pub fn try_rewrite_cancel_same_base_powers_div_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CancelSameBasePowersRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    let (base_num, exp_num) = as_pow(ctx, num)?;
    let (base_den, exp_den) = as_pow(ctx, den)?;

    if compare_expr(ctx, base_num, base_den) != std::cmp::Ordering::Equal {
        return None;
    }

    let m = as_i64(ctx, exp_num)?;
    let n = as_i64(ctx, exp_den)?;
    if m == 0 && n == 0 {
        return None;
    }

    let diff = m - n;
    let (rewritten, kind) = if diff == 0 {
        (ctx.num(1), CancelSameBasePowersRewriteKind::EqualPowers)
    } else if diff == 1 {
        (base_num, CancelSameBasePowersRewriteKind::CollapseToBase)
    } else if diff == -1 {
        let one = ctx.num(1);
        let rewritten = ctx.add(Expr::Div(one, base_num));
        (
            rewritten,
            CancelSameBasePowersRewriteKind::CollapseToReciprocalBase,
        )
    } else if diff > 0 {
        let new_exp = ctx.num(diff);
        let rewritten = ctx.add(Expr::Pow(base_num, new_exp));
        (
            rewritten,
            CancelSameBasePowersRewriteKind::CollapseToPositivePower(diff),
        )
    } else {
        let pos_diff = -diff;
        let new_exp = ctx.num(pos_diff);
        let pow_result = ctx.add(Expr::Pow(base_num, new_exp));
        let one = ctx.num(1);
        let rewritten = ctx.add(Expr::Div(one, pow_result));
        (
            rewritten,
            CancelSameBasePowersRewriteKind::CollapseToReciprocalPower(pos_diff),
        )
    };

    Some(CancelSameBasePowersRewrite {
        rewritten,
        nonzero_target: base_num,
        kind,
    })
}

/// Try to rewrite `P/P` as `1`.
pub fn try_rewrite_cancel_identical_fraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CancelIdenticalFractionRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    if compare_expr(ctx, num, den) != std::cmp::Ordering::Equal {
        return None;
    }

    Some(CancelIdenticalFractionRewrite {
        rewritten: ctx.num(1),
        nonzero_target: den,
        kind: CancelIdenticalFractionRewriteKind::IdenticalFraction,
    })
}

/// Try to rewrite `P^n / P` or `P^n / (-P)` for integer `n >= 1`.
pub fn try_rewrite_cancel_power_fraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CancelPowerFractionRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    let (base, exp) = as_pow(ctx, num)?;
    let relation = poly_relation(ctx, base, den)?;
    let exp_val = as_i64(ctx, exp)?;
    if exp_val < 1 {
        return None;
    }

    let base_result = if exp_val == 1 {
        ctx.num(1)
    } else if exp_val == 2 {
        base
    } else {
        let new_exp = ctx.num(exp_val - 1);
        ctx.add(Expr::Pow(base, new_exp))
    };

    let (rewritten, kind) = match relation {
        SignRelation::Same => (base_result, CancelPowerFractionRewriteKind::SameSign),
        SignRelation::Negated => {
            let negated = ctx.add(Expr::Neg(base_result));
            (negated, CancelPowerFractionRewriteKind::NegatedDenominator)
        }
    };

    Some(CancelPowerFractionRewrite {
        rewritten,
        nonzero_target: den,
        kind,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_cancel_identical_fraction_expr, try_rewrite_cancel_power_fraction_expr,
        try_rewrite_cancel_same_base_powers_div_expr,
    };
    use crate::poly_compare::poly_eq;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_ast::Expr;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn cancel_same_base_powers_rewrites_positive_diff() {
        let mut ctx = Context::new();
        let expr = parse("x^5 / x^2", &mut ctx).expect("parse");
        let rw = try_rewrite_cancel_same_base_powers_div_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("x^3", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, rw.rewritten, expected));
    }

    #[test]
    fn cancel_same_base_powers_rewrites_negative_diff() {
        let mut ctx = Context::new();
        let expr = parse("x^2 / x^5", &mut ctx).expect("parse");
        let rw = try_rewrite_cancel_same_base_powers_div_expr(&mut ctx, expr).expect("rewrite");
        let x = parse("x", &mut ctx).expect("x");
        let exp = ctx.num(3);
        let one = ctx.num(1);
        let pow = ctx.add(Expr::Pow(x, exp));
        let expected = ctx.add(Expr::Div(one, pow));
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn cancel_same_base_powers_rejects_non_matching_bases() {
        let mut ctx = Context::new();
        let expr = parse("x^2 / y^2", &mut ctx).expect("parse");
        assert!(try_rewrite_cancel_same_base_powers_div_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn cancel_identical_fraction_rewrites_to_one() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)/(x+1)", &mut ctx).expect("parse");
        let rw = try_rewrite_cancel_identical_fraction_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("1", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, rw.rewritten, expected));
    }

    #[test]
    fn cancel_power_fraction_rewrites_same_sign() {
        let mut ctx = Context::new();
        let expr = parse("x^4/x", &mut ctx).expect("parse");
        let rw = try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("x^3", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, rw.rewritten, expected));
    }

    #[test]
    fn cancel_power_fraction_rewrites_negated_denominator() {
        let mut ctx = Context::new();
        let expr = parse("x^3/(-x)", &mut ctx).expect("parse");
        let rw = try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("-x^2", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, rw.rewritten, expected));
    }

    #[test]
    fn cancel_power_fraction_rejects_symbolic_exponent_without_poly_compare() {
        let mut ctx = Context::new();
        let expr = parse("(a^x)/a", &mut ctx).expect("parse");
        assert!(try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).is_none());
    }
}

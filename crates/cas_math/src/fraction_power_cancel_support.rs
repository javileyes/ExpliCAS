//! Support for power/fraction cancellation rewrites.
//!
//! This module contains structural/algebraic rewrite planning for:
//! - `P^m / P^n -> P^(m-n)` (integer exponents)
//! - `P / P -> 1`
//! - `P^n / P -> P^(n-1)` and `P^n / (-P) -> -P^(n-1)`
//!
//! Domain-mode policy (strict/assume) is intentionally left to callers.

use crate::expr_destructure::{as_div, as_pow};
use crate::expr_nary::{build_balanced_mul, mul_leaves};
use crate::expr_predicates::is_even_root_exponent;
use crate::multipoly::{multipoly_from_expr, PolyBudget};
use crate::numeric::as_i64;
use crate::numeric_eval::as_rational_const;
use crate::poly_compare::{poly_relation, SignRelation};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

#[inline]
fn expr_matches(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    lhs == rhs || compare_expr(ctx, lhs, rhs) == std::cmp::Ordering::Equal
}

fn relation_to_base(ctx: &Context, base: ExprId, den: ExprId) -> Option<SignRelation> {
    if expr_matches(ctx, base, den) {
        Some(SignRelation::Same)
    } else if let Expr::Neg(inner) = ctx.get(den) {
        if expr_matches(ctx, base, *inner) {
            Some(SignRelation::Negated)
        } else {
            poly_relation(ctx, base, den)
        }
    } else {
        poly_relation(ctx, base, den)
    }
}

fn compare_budget() -> PolyBudget {
    PolyBudget {
        max_terms: 100,
        max_total_degree: 10,
        max_pow_exp: 5,
    }
}

fn scalar_multiple_to_base(ctx: &mut Context, base: ExprId, expr: ExprId) -> Option<BigRational> {
    if expr_matches(ctx, base, expr) {
        return Some(BigRational::one());
    }

    if let Expr::Neg(inner) = ctx.get(expr) {
        if expr_matches(ctx, base, *inner) {
            return Some(-BigRational::one());
        }
    }

    let (structural_scale, structural_core) = split_numeric_scale(ctx, expr);
    if !structural_scale.is_zero() && expr_matches(ctx, base, structural_core) {
        return Some(structural_scale);
    }

    let budget = compare_budget();
    let base_poly = multipoly_from_expr(ctx, base, &budget).ok()?;
    let expr_poly = multipoly_from_expr(ctx, expr, &budget).ok()?;
    let base_content = base_poly.content();
    let expr_content = expr_poly.content();
    if base_content.is_zero() || expr_content.is_zero() {
        return None;
    }

    let (_, base_primitive) = base_poly.primitive_part();
    let (_, expr_primitive) = expr_poly.primitive_part();
    if expr_primitive == base_primitive {
        Some(expr_content / base_content)
    } else if expr_primitive == base_primitive.neg() {
        Some(-(expr_content / base_content))
    } else {
        None
    }
}

fn split_numeric_scale(ctx: &mut Context, expr: ExprId) -> (BigRational, ExprId) {
    if let Some(value) = as_rational_const(ctx, expr) {
        return (value, ctx.num(1));
    }

    if let Expr::Neg(inner) = ctx.get(expr) {
        let (scale, core) = split_numeric_scale(ctx, *inner);
        return (-scale, core);
    }

    let mut scale = BigRational::one();
    let mut core_factors = Vec::new();
    for factor in mul_leaves(ctx, expr) {
        if let Some(value) = as_rational_const(ctx, factor) {
            scale *= value;
        } else {
            core_factors.push(factor);
        }
    }

    (scale, build_balanced_mul(ctx, &core_factors))
}

fn scale_rewrite_result(ctx: &mut Context, scale: BigRational, expr: ExprId) -> ExprId {
    if scale.is_one() {
        expr
    } else if scale == -BigRational::one() {
        ctx.add(Expr::Neg(expr))
    } else if matches!(ctx.get(expr), Expr::Number(n) if n.is_one()) {
        ctx.add(Expr::Number(scale))
    } else {
        let scale_expr = ctx.add(Expr::Number(scale));
        ctx.add(Expr::Mul(scale_expr, expr))
    }
}

fn is_one_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn build_product_or_one(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    if factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, factors)
    }
}

fn scaled_power_factor_from_numerator(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<(BigRational, ExprId, ExprId, ExprId, BigRational)> {
    let (num_scale, num_core) = split_numeric_scale(ctx, num);
    if let Some((base, exp)) = as_pow(ctx, num_core) {
        let den_scale = scalar_multiple_to_base(ctx, base, den).or_else(|| {
            relation_to_base(ctx, base, den).map(|relation| match relation {
                SignRelation::Same => BigRational::one(),
                SignRelation::Negated => -BigRational::one(),
            })
        })?;
        return Some((num_scale, ctx.num(1), base, exp, den_scale));
    }

    let factors = mul_leaves(ctx, num_core);
    for (idx, factor) in factors.iter().enumerate() {
        let Some((base, exp)) = as_pow(ctx, *factor) else {
            continue;
        };
        let Some(den_scale) = scalar_multiple_to_base(ctx, base, den).or_else(|| {
            relation_to_base(ctx, base, den).map(|relation| match relation {
                SignRelation::Same => BigRational::one(),
                SignRelation::Negated => -BigRational::one(),
            })
        }) else {
            continue;
        };

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        let cofactor = build_product_or_one(ctx, &cofactor_factors);
        return Some((num_scale, cofactor, base, exp, den_scale));
    }

    None
}

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

#[derive(Debug, Clone)]
pub struct CollapseReciprocalNegativePowerRewrite {
    pub rewritten: ExprId,
    pub nonzero_target: ExprId,
}

/// Try to rewrite `P^m / P^n` into a simpler power form (integer exponents).
pub fn try_rewrite_cancel_same_base_powers_div_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CancelSameBasePowersRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    let (base_num, exp_num) = as_pow(ctx, num)?;
    let (base_den, exp_den) = as_pow(ctx, den)?;

    if !expr_matches(ctx, base_num, base_den) {
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

/// Try to rewrite `P^n / P` or `P^n / (-P)` for integer `n >= 1`, plus
/// positive even-root exponents such as `P^(1/2) / (-P)`.
pub fn try_rewrite_cancel_power_fraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CancelPowerFractionRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    let (num_scale, cofactor, base, exp, den_scale) =
        scaled_power_factor_from_numerator(ctx, num, den)?;
    if den_scale.is_zero() {
        return None;
    }
    let result_scale = num_scale / den_scale.clone();

    if let Some(exp_val) = as_rational_const(ctx, exp) {
        if !exp_val.is_integer() && is_even_root_exponent(&exp_val) {
            let new_exp_val = exp_val - num_rational::BigRational::one();
            let base_result = if new_exp_val.is_zero() {
                ctx.num(1)
            } else if new_exp_val.is_one() {
                base
            } else {
                let new_exp = ctx.add(Expr::Number(new_exp_val));
                ctx.add(Expr::Pow(base, new_exp))
            };

            let scaled = scale_rewrite_result(ctx, result_scale, base_result);
            let rewritten = if is_one_expr(ctx, cofactor) {
                scaled
            } else {
                ctx.add(Expr::Mul(cofactor, scaled))
            };
            let kind = if den_scale.is_positive() {
                CancelPowerFractionRewriteKind::SameSign
            } else {
                CancelPowerFractionRewriteKind::NegatedDenominator
            };

            return Some(CancelPowerFractionRewrite {
                rewritten,
                nonzero_target: base,
                kind,
            });
        }
    }

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

    let scaled = scale_rewrite_result(ctx, result_scale, base_result);
    let rewritten = if is_one_expr(ctx, cofactor) {
        scaled
    } else {
        ctx.add(Expr::Mul(cofactor, scaled))
    };
    let kind = if den_scale.is_positive() {
        CancelPowerFractionRewriteKind::SameSign
    } else {
        CancelPowerFractionRewriteKind::NegatedDenominator
    };

    Some(CancelPowerFractionRewrite {
        rewritten,
        nonzero_target: base,
        kind,
    })
}

/// Try to rewrite `1 / (P^(-a))` into `P^a`.
///
/// This requires `P != 0` to preserve the original domain, so callers should
/// attach an explicit nonzero condition when using the rewrite in generic/assume
/// modes.
pub fn try_rewrite_collapse_reciprocal_negative_power_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CollapseReciprocalNegativePowerRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    let Expr::Number(numerator) = ctx.get(num) else {
        return None;
    };
    if !numerator.is_one() {
        return None;
    }

    let (base, exp) = as_pow(ctx, den)?;
    let Expr::Number(exp_num) = ctx.get(exp) else {
        return None;
    };
    if !exp_num.is_negative() {
        return None;
    }

    let positive_exp = ctx.add(Expr::Number(-exp_num.clone()));
    let rewritten = if matches!(ctx.get(positive_exp), Expr::Number(n) if n.is_one()) {
        base
    } else {
        ctx.add(Expr::Pow(base, positive_exp))
    };

    Some(CollapseReciprocalNegativePowerRewrite {
        rewritten,
        nonzero_target: base,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_cancel_identical_fraction_expr, try_rewrite_cancel_power_fraction_expr,
        try_rewrite_cancel_same_base_powers_div_expr,
        try_rewrite_collapse_reciprocal_negative_power_expr,
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

    #[test]
    fn cancel_power_fraction_rewrites_negated_even_root_polynomial_orientation() {
        let mut ctx = Context::new();
        let base = parse("1-x^2", &mut ctx).expect("parse base");
        let den = parse("x^2-1", &mut ctx).expect("parse denominator");
        let half = ctx.rational(1, 2);
        let num = ctx.add(Expr::Pow(base, half));
        let expr = ctx.add(Expr::Div(num, den));
        let rw = try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).expect("rewrite");
        let rendered = cas_formatter::render_expr(&ctx, rw.rewritten);

        assert!(rendered.starts_with("-"), "rendered={rendered}");
        assert!(rendered.contains("(1 - x^2)^(-1/2)"), "rendered={rendered}");
    }

    #[test]
    fn cancel_power_fraction_rewrites_scaled_even_root_quotient() {
        let mut ctx = Context::new();
        let u = parse("u", &mut ctx).expect("parse u");
        let half = ctx.rational(1, 2);
        let pow = ctx.add(Expr::Pow(u, half));
        let neg_two = ctx.num(-2);
        let num = ctx.add(Expr::Mul(neg_two, pow));
        let den = parse("2*u", &mut ctx).expect("parse denominator");
        let expr = ctx.add(Expr::Div(num, den));
        let rw = try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).expect("rewrite");
        let rendered = cas_formatter::render_expr(&ctx, rw.rewritten);

        assert!(rendered.starts_with("-"), "rendered={rendered}");
        assert!(rendered.contains("u^(-1/2)"), "rendered={rendered}");
    }

    #[test]
    fn cancel_power_fraction_rewrites_scaled_opaque_even_root_denominator() {
        let mut ctx = Context::new();
        let base = parse("sin(x)", &mut ctx).expect("parse base");
        let half = ctx.rational(1, 2);
        let num = ctx.add(Expr::Pow(base, half));
        let den = parse("2*sin(x)", &mut ctx).expect("parse denominator");
        let expr = ctx.add(Expr::Div(num, den));
        let rw = try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).expect("rewrite");
        let rendered = cas_formatter::render_expr(&ctx, rw.rewritten);

        assert!(rendered.contains("1/2"), "rendered={rendered}");
        assert!(rendered.contains("sin(x)^(-1/2)"), "rendered={rendered}");
    }

    #[test]
    fn cancel_power_fraction_rewrites_cofactored_even_root_quotient() {
        let mut ctx = Context::new();
        let base = parse("u^2 + 1", &mut ctx).expect("parse base");
        let x = parse("x", &mut ctx).expect("parse x");
        let half = ctx.rational(1, 2);
        let pow = ctx.add(Expr::Pow(base, half));
        let num = ctx.add(Expr::Mul(x, pow));
        let expr = ctx.add(Expr::Div(num, base));
        let rw = try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).expect("rewrite");
        let rendered = cas_formatter::render_expr(&ctx, rw.rewritten);

        assert!(rendered.contains("x"), "rendered={rendered}");
        assert!(rendered.contains("(u^2 + 1)^(-1/2)"), "rendered={rendered}");
    }

    #[test]
    fn cancel_power_fraction_rewrites_scaled_polynomial_denominator() {
        let mut ctx = Context::new();
        let base = parse("-x^2-x", &mut ctx).expect("parse base");
        let half = ctx.rational(1, 2);
        let pow = ctx.add(Expr::Pow(base, half));
        let neg_two = ctx.num(-2);
        let num = ctx.add(Expr::Mul(neg_two, pow));
        let den = parse("-2*x^2-2*x", &mut ctx).expect("parse denominator");
        let expr = ctx.add(Expr::Div(num, den));
        let rw = try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).expect("rewrite");
        let rendered = cas_formatter::render_expr(&ctx, rw.rewritten);

        assert!(rendered.starts_with("-"), "rendered={rendered}");
        assert!(rendered.contains("(-1/2)"), "rendered={rendered}");
    }

    #[test]
    fn cancel_power_fraction_rewrites_negative_even_root_quotient() {
        let mut ctx = Context::new();
        let expr = parse("(2*x^2+2*x-3)^(-1/2)/(2*x^2+2*x-3)", &mut ctx).expect("parse");
        let rw = try_rewrite_cancel_power_fraction_expr(&mut ctx, expr).expect("rewrite");
        let rendered = cas_formatter::render_expr(&ctx, rw.rewritten);

        assert!(
            rendered.contains("(2 * x^2 + 2 * x - 3)^(-3/2)"),
            "rendered={rendered}"
        );
    }

    #[test]
    fn collapse_reciprocal_negative_power_rewrites_fractional_exponent() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let neg_half = ctx.add(Expr::Number(num_rational::BigRational::new(
            (-1).into(),
            2.into(),
        )));
        let pow = ctx.add(Expr::Pow(x, neg_half));
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Div(one, pow));
        let rw =
            try_rewrite_collapse_reciprocal_negative_power_expr(&mut ctx, expr).expect("rewrite");
        let pos_half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let expected = ctx.add(Expr::Pow(x, pos_half));
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn collapse_reciprocal_negative_power_rewrites_integer_exponent() {
        let mut ctx = Context::new();
        let expr = parse("1/(x^(-1))", &mut ctx).expect("parse");
        let rw =
            try_rewrite_collapse_reciprocal_negative_power_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("x", &mut ctx).expect("expected");
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }
}

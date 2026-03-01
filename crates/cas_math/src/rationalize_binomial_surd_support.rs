//! Support for rationalizing binomial denominators with a single numeric surd.

use crate::build::mul2_raw;
use crate::expr_destructure::as_div;
use crate::root_forms::extract_numeric_sqrt_radicand;
use cas_ast::views::{as_rational_const, count_distinct_numeric_surds, is_surd_free};
use cas_ast::{count_nodes, Context, Expr, ExprId};
use num_rational::BigRational;

#[derive(Debug, Clone, Copy)]
pub struct RationalizeBinomialSurdRewrite {
    pub rewritten: ExprId,
    pub num: ExprId,
    pub den: ExprId,
    pub new_num: ExprId,
    pub new_den: ExprId,
}

struct BinomialSurd {
    a: BigRational,
    b: BigRational,
    n: i64,
    is_sub: bool,
}

/// Try to rationalize a denominator of form `A ± B*sqrt(n)` (optionally multiplied
/// by surd-free factors).
pub fn try_rewrite_rationalize_binomial_surd_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RationalizeBinomialSurdRewrite> {
    let (num, den) = as_div(ctx, expr)?;

    if count_nodes(ctx, den) > 30 {
        return None;
    }

    let distinct_surds = count_distinct_numeric_surds(ctx, den, 50);
    if distinct_surds != 1 {
        return None;
    }

    let (k_factors, surd) = extract_binomial_from_product(ctx, den)?;
    let k_factor = multiply_factors(ctx, &k_factors);

    let a_sq = &surd.a * &surd.a;
    let b_sq = &surd.b * &surd.b;
    let b_sq_n = &b_sq * BigRational::from_integer(surd.n.into());
    let new_den_val = &a_sq - &b_sq_n;
    if new_den_val == BigRational::from_integer(0.into()) {
        return None;
    }

    let a_expr = ctx.add(Expr::Number(surd.a.clone()));
    let n_expr = ctx.num(surd.n);
    let half = ctx.rational(1, 2);
    let sqrt_n = ctx.add(Expr::Pow(n_expr, half));
    let b_sqrt_n = if surd.b == BigRational::from_integer(1.into()) {
        sqrt_n
    } else if surd.b == BigRational::from_integer((-1).into()) {
        ctx.add(Expr::Neg(sqrt_n))
    } else {
        let b_expr = ctx.add(Expr::Number(surd.b.clone()));
        mul2_raw(ctx, b_expr, sqrt_n)
    };

    let conjugate = if surd.is_sub {
        ctx.add(Expr::Add(a_expr, b_sqrt_n))
    } else {
        ctx.add(Expr::Sub(a_expr, b_sqrt_n))
    };

    let (final_conjugate, final_den_val) = if new_den_val < BigRational::from_integer(0.into()) {
        (ctx.add(Expr::Neg(conjugate)), -new_den_val.clone())
    } else {
        (conjugate, new_den_val.clone())
    };

    let new_num = mul2_raw(ctx, num, final_conjugate);
    let new_den = ctx.add(Expr::Number(final_den_val.clone()));

    let rewritten = if final_den_val == BigRational::from_integer(1.into()) {
        match k_factor {
            Some(k) => ctx.add(Expr::Div(new_num, k)),
            None => new_num,
        }
    } else {
        let rationalized_den = ctx.add(Expr::Number(final_den_val.clone()));
        let full_den = match k_factor {
            Some(k) => mul2_raw(ctx, k, rationalized_den),
            None => rationalized_den,
        };
        ctx.add(Expr::Div(new_num, full_den))
    };

    if count_nodes(ctx, rewritten) > count_nodes(ctx, expr) + 20 {
        return None;
    }

    Some(RationalizeBinomialSurdRewrite {
        rewritten,
        num,
        den,
        new_num,
        new_den,
    })
}

/// Build a user-facing description for binomial-surd rationalization.
///
/// Caller provides expression rendering to keep this module independent from
/// formatter crates.
pub fn format_rationalize_binomial_surd_desc_with<FRender>(
    rewrite: RationalizeBinomialSurdRewrite,
    mut render_expr: FRender,
) -> String
where
    FRender: FnMut(ExprId) -> String,
{
    format!(
        "{} / {} -> {} / {}",
        render_expr(rewrite.num),
        render_expr(rewrite.den),
        render_expr(rewrite.new_num),
        render_expr(rewrite.new_den)
    )
}

/// Rewrite helper returning `(rewritten_expr, description)`.
pub fn rewrite_rationalize_binomial_surd_expr_with<FRender>(
    ctx: &mut Context,
    expr: ExprId,
    mut render_expr: FRender,
) -> Option<(ExprId, String)>
where
    FRender: FnMut(&Context, ExprId) -> String,
{
    let rewrite = try_rewrite_rationalize_binomial_surd_expr(ctx, expr)?;
    let desc = format_rationalize_binomial_surd_desc_with(rewrite, |id| render_expr(ctx, id));
    Some((rewrite.rewritten, desc))
}

fn parse_binomial_surd(ctx: &Context, den: ExprId) -> Option<BinomialSurd> {
    fn parse_surd_term(ctx: &Context, id: ExprId) -> Option<(BigRational, i64)> {
        if let Expr::Neg(inner) = ctx.get(id) {
            let (b, n) = parse_surd_term(ctx, *inner)?;
            return Some((-b, n));
        }
        if let Some(n) = extract_numeric_sqrt_radicand(ctx, id) {
            return Some((BigRational::from_integer(1.into()), n));
        }
        if let Expr::Mul(l, r) = ctx.get(id) {
            if let Some(n) = extract_numeric_sqrt_radicand(ctx, *r) {
                if let Some(b) = as_rational_const(ctx, *l, 8) {
                    return Some((b, n));
                }
            }
            if let Some(n) = extract_numeric_sqrt_radicand(ctx, *l) {
                if let Some(b) = as_rational_const(ctx, *r, 8) {
                    return Some((b, n));
                }
            }
        }
        None
    }

    match ctx.get(den) {
        Expr::Add(l, r) => {
            if let Some(a) = as_rational_const(ctx, *l, 8) {
                if let Some((b, n)) = parse_surd_term(ctx, *r) {
                    return Some(BinomialSurd {
                        a,
                        b,
                        n,
                        is_sub: false,
                    });
                }
            }
            if let Some(a) = as_rational_const(ctx, *r, 8) {
                if let Some((b, n)) = parse_surd_term(ctx, *l) {
                    return Some(BinomialSurd {
                        a,
                        b,
                        n,
                        is_sub: false,
                    });
                }
            }
            None
        }
        Expr::Sub(l, r) => {
            if let Some(a) = as_rational_const(ctx, *l, 8) {
                if let Some((b, n)) = parse_surd_term(ctx, *r) {
                    return Some(BinomialSurd {
                        a,
                        b,
                        n,
                        is_sub: true,
                    });
                }
            }
            if let Some(a) = as_rational_const(ctx, *r, 8) {
                if let Some((b, n)) = parse_surd_term(ctx, *l) {
                    return Some(BinomialSurd {
                        a: -a,
                        b,
                        n,
                        is_sub: false,
                    });
                }
            }
            None
        }
        _ => None,
    }
}

fn extract_binomial_from_product(
    ctx: &Context,
    den: ExprId,
) -> Option<(Vec<ExprId>, BinomialSurd)> {
    if let Some(surd) = parse_binomial_surd(ctx, den) {
        return Some((vec![], surd));
    }

    let Expr::Mul(_, _) = ctx.get(den) else {
        return None;
    };

    let mut factors = Vec::new();
    collect_factors(ctx, den, &mut factors);

    let mut binomial_idx = None;
    for (i, &factor) in factors.iter().enumerate() {
        if parse_binomial_surd(ctx, factor).is_some() {
            if binomial_idx.is_some() {
                return None;
            }
            binomial_idx = Some(i);
        } else if !is_surd_free(ctx, factor, 20) {
            return None;
        }
    }

    let binomial_idx = binomial_idx?;
    let binomial = parse_binomial_surd(ctx, factors[binomial_idx])?;
    let k_factors: Vec<_> = factors
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i != binomial_idx)
        .map(|(_, f)| f)
        .collect();
    Some((k_factors, binomial))
}

fn collect_factors(ctx: &Context, id: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(id) {
        Expr::Mul(l, r) => {
            collect_factors(ctx, *l, factors);
            collect_factors(ctx, *r, factors);
        }
        _ => factors.push(id),
    }
}

fn multiply_factors(ctx: &mut Context, factors: &[ExprId]) -> Option<ExprId> {
    if factors.is_empty() {
        return None;
    }
    let mut acc = factors[0];
    for &f in &factors[1..] {
        acc = ctx.add(Expr::Mul(acc, f));
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::{
        format_rationalize_binomial_surd_desc_with, rewrite_rationalize_binomial_surd_expr_with,
        try_rewrite_rationalize_binomial_surd_expr,
    };
    use cas_ast::views::{count_distinct_numeric_surds, is_surd_free};
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn rationalizes_single_binomial_surd() {
        let mut ctx = Context::new();
        let expr = parse("x/(3+sqrt(2))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_binomial_surd_expr(&mut ctx, expr).expect("rewrite");
        if let Expr::Div(_num, den) = ctx.get(rewrite.rewritten) {
            assert!(is_surd_free(&ctx, *den, 40));
            assert_eq!(count_distinct_numeric_surds(&ctx, *den, 40), 0);
        } else {
            panic!("expected division output");
        }
    }

    #[test]
    fn skips_multi_surd_denominator() {
        let mut ctx = Context::new();
        let expr = parse("x/(sqrt(2)+sqrt(3))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_binomial_surd_expr(&mut ctx, expr);
        assert!(rewrite.is_none());
    }

    #[test]
    fn format_desc_includes_arrow() {
        let mut ctx = Context::new();
        let expr = parse("x/(3+sqrt(2))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_binomial_surd_expr(&mut ctx, expr).expect("rewrite");
        let desc = format_rationalize_binomial_surd_desc_with(rewrite, |id| format!("{:?}", id));
        assert!(desc.contains("->"));
    }

    #[test]
    fn rewrite_with_returns_expr_and_desc() {
        let mut ctx = Context::new();
        let expr = parse("x/(3+sqrt(2))", &mut ctx).expect("parse");
        let out = rewrite_rationalize_binomial_surd_expr_with(&mut ctx, expr, |_core_ctx, id| {
            format!("{:?}", id)
        })
        .expect("rewrite");
        assert!(out.1.contains("->"));
        assert_ne!(out.0, expr);
    }
}

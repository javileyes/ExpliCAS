//! Support rewrites for products/quotients involving difference factors.

use crate::build::mul2_raw;
use crate::difference_factor_support::{build_difference_expr, extract_difference_pair};
use crate::fraction_factors::collect_mul_factors_flat as collect_mul_factors;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_traits::Signed;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct AbsorbNegationIntoDifferenceRewrite {
    pub rewritten: ExprId,
}

#[derive(Debug, Clone, Copy)]
pub struct CanonicalDifferenceProductRewrite {
    pub rewritten: ExprId,
}

/// Try to absorb a leading negation in a rational expression into one
/// denominator difference factor by flipping `(x-y) -> (y-x)`.
pub fn try_rewrite_absorb_negation_into_difference_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<AbsorbNegationIntoDifferenceRewrite> {
    let (is_neg_wrapped, div_num, div_den) = match ctx.get(expr) {
        Expr::Neg(inner) => {
            if let Expr::Div(n, d) = ctx.get(*inner) {
                (true, *n, *d)
            } else {
                return None;
            }
        }
        Expr::Div(n, d) => {
            if let Expr::Number(num_val) = ctx.get(*n) {
                if num_val.is_negative() {
                    (false, *n, *d)
                } else {
                    return None;
                }
            } else if let Expr::Neg(_) = ctx.get(*n) {
                (false, *n, *d)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let mut factors: Vec<ExprId> = collect_mul_factors(ctx, div_den);
    let mut flip_index: Option<usize> = None;
    let mut diff_pair: Option<(ExprId, ExprId)> = None;
    for (i, &f) in factors.iter().enumerate() {
        if let Some((x, y)) = extract_difference_pair(ctx, f) {
            flip_index = Some(i);
            diff_pair = Some((x, y));
            break;
        }
    }

    let idx = flip_index?;
    let (x, y) = diff_pair?;
    factors[idx] = build_difference_expr(ctx, y, x);

    let new_den = factors.iter().copied().fold(None, |acc, f| {
        Some(match acc {
            Some(a) => mul2_raw(ctx, a, f),
            None => f,
        })
    })?;

    let new_num = if is_neg_wrapped {
        div_num
    } else if let Expr::Number(n) = ctx.get(div_num) {
        ctx.add(Expr::Number(-n.clone()))
    } else if let Expr::Neg(inner) = ctx.get(div_num) {
        *inner
    } else {
        return None;
    };

    Some(AbsorbNegationIntoDifferenceRewrite {
        rewritten: ctx.add(Expr::Div(new_num, new_den)),
    })
}

/// Canonicalize `1/((p-t)*(q-t))` into `1/((t-p)*(t-q))` when two difference
/// factors share the same tail.
pub fn try_rewrite_canonical_difference_product_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CanonicalDifferenceProductRewrite> {
    let (num, den) = if let Expr::Div(n, d) = ctx.get(expr) {
        (*n, *d)
    } else {
        return None;
    };

    let (factor1, factor2) = if let Expr::Mul(l, r) = ctx.get(den) {
        (*l, *r)
    } else {
        return None;
    };

    let (p, t1) = if let Expr::Sub(a, b) = ctx.get(factor1) {
        (*a, *b)
    } else {
        return None;
    };
    let (q, t2) = if let Expr::Sub(a, b) = ctx.get(factor2) {
        (*a, *b)
    } else {
        return None;
    };

    if compare_expr(ctx, t1, t2) != Ordering::Equal {
        return None;
    }
    let t = t1;

    let t_already_first_1 = if let Expr::Sub(a, _) = ctx.get(factor1) {
        compare_expr(ctx, *a, t) == Ordering::Equal
    } else {
        false
    };
    let t_already_first_2 = if let Expr::Sub(a, _) = ctx.get(factor2) {
        compare_expr(ctx, *a, t) == Ordering::Equal
    } else {
        false
    };
    if t_already_first_1 && t_already_first_2 {
        return None;
    }

    let new_factor1 = ctx.add(Expr::Sub(t, p));
    let new_factor2 = ctx.add(Expr::Sub(t, q));
    let new_den = mul2_raw(ctx, new_factor1, new_factor2);
    Some(CanonicalDifferenceProductRewrite {
        rewritten: ctx.add(Expr::Div(num, new_den)),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_absorb_negation_into_difference_expr,
        try_rewrite_canonical_difference_product_expr,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn absorbs_negation_into_difference_factor() {
        let mut ctx = Context::new();
        let expr = parse("-1/((x-y)*z)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_absorb_negation_into_difference_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }

    #[test]
    fn canonicalizes_same_tail_difference_product() {
        let mut ctx = Context::new();
        let expr = parse("1/((p-t)*(q-t))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_canonical_difference_product_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }
}

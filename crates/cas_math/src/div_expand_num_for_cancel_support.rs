//! Support for expanding a multiplicative numerator factor to expose cancelable
//! common factors against an additive denominator.

use crate::build::mul2_raw;
use crate::expr_destructure::as_div;
use crate::fraction_factors::collect_mul_factors_int_pow;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct DivExpandNumForCancelRewrite {
    pub rewritten: ExprId,
}

/// Try to expand one additive factor inside the numerator product when that
/// expansion creates terms sharing symbolic factors with denominator Add terms.
pub fn try_rewrite_div_expand_num_for_cancel_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DivExpandNumForCancelRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    if !matches!(ctx.get(den), Expr::Add(_, _)) {
        return None;
    }
    if matches!(ctx.get(num), Expr::Add(_, _)) {
        return None;
    }

    let num_mul_factors = collect_mul_factors_flat(ctx, num);
    if num_mul_factors.len() < 2 {
        return None;
    }

    let mut den_terms = Vec::new();
    collect_add_terms(ctx, den, &mut den_terms);
    if den_terms.is_empty() {
        return None;
    }

    let mut den_common = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, den_terms[0]));
    for &term_id in den_terms.iter().skip(1) {
        let term_map = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, term_id));
        den_common.retain_mut(|(base, exp)| {
            if let Some(term_exp) = find_factor_exp(ctx, &term_map, *base) {
                *exp = (*exp).min(term_exp);
                *exp >= 1
            } else {
                false
            }
        });
        if den_common.is_empty() {
            return None;
        }
    }

    for (add_idx, &candidate) in num_mul_factors.iter().enumerate() {
        if !matches!(ctx.get(candidate), Expr::Add(_, _)) {
            continue;
        }

        let outer_factors: Vec<ExprId> = num_mul_factors
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != add_idx)
            .map(|(_, &f)| f)
            .collect();
        if outer_factors.is_empty() {
            continue;
        }

        let mut outer = outer_factors[0];
        for &f in outer_factors.iter().skip(1) {
            outer = mul2_raw(ctx, outer, f);
        }

        let mut add_terms = Vec::new();
        collect_add_terms(ctx, candidate, &mut add_terms);
        if add_terms.len() < 2 {
            continue;
        }

        let expanded_terms: Vec<ExprId> =
            add_terms.iter().map(|&t| mul2_raw(ctx, outer, t)).collect();
        let mut num_common =
            factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, expanded_terms[0]));
        let mut bail = false;
        for &term_id in expanded_terms.iter().skip(1) {
            let term_map = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, term_id));
            num_common.retain_mut(|(base, exp)| {
                if let Some(te) = find_factor_exp(ctx, &term_map, *base) {
                    *exp = (*exp).min(te);
                    *exp >= 1
                } else {
                    false
                }
            });
            if num_common.is_empty() {
                bail = true;
                break;
            }
        }
        if bail {
            continue;
        }

        if !has_shared_factor(ctx, &num_common, &den_common) {
            continue;
        }

        let mut new_num = expanded_terms[0];
        for &term in expanded_terms.iter().skip(1) {
            new_num = ctx.add(Expr::Add(new_num, term));
        }
        let rewritten = ctx.add(Expr::Div(new_num, den));
        if rewritten != expr {
            return Some(DivExpandNumForCancelRewrite { rewritten });
        }
    }

    None
}

fn collect_mul_factors_flat(ctx: &Context, num: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    let mut stack = vec![num];
    while let Some(curr) = stack.pop() {
        if let Expr::Mul(l, r) = ctx.get(curr) {
            stack.push(*r);
            stack.push(*l);
        } else {
            factors.push(curr);
        }
    }
    factors
}

fn collect_add_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_add_terms(ctx, *l, terms);
            collect_add_terms(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}

fn factors_to_vec(ctx: &Context, factors: &[(ExprId, i64)]) -> Vec<(ExprId, i64)> {
    let mut out: Vec<(ExprId, i64)> = Vec::new();
    for &(base, exp) in factors {
        if matches!(ctx.get(base), Expr::Number(_)) {
            continue;
        }
        if let Some((_, total_exp)) = out
            .iter_mut()
            .find(|(existing, _)| compare_expr(ctx, *existing, base) == Ordering::Equal)
        {
            *total_exp += exp;
        } else {
            out.push((base, exp));
        }
    }
    out
}

fn find_factor_exp(ctx: &Context, factors: &[(ExprId, i64)], base: ExprId) -> Option<i64> {
    factors
        .iter()
        .find(|(b, _)| compare_expr(ctx, *b, base) == Ordering::Equal)
        .map(|(_, exp)| *exp)
}

fn has_shared_factor(ctx: &Context, a: &[(ExprId, i64)], b: &[(ExprId, i64)]) -> bool {
    a.iter().any(|(base, _)| {
        b.iter()
            .any(|(other, _)| compare_expr(ctx, *base, *other) == Ordering::Equal)
    })
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_div_expand_num_for_cancel_expr;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn expands_numerator_product_when_factor_overlap_exists() {
        let mut ctx = Context::new();
        let expr = parse("(y*(x*a+x*b))/(x*c+x*d)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_div_expand_num_for_cancel_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }
}

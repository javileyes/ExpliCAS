//! Support for cancelling common symbolic factors in Add/Add fractions.

use crate::expr_destructure::as_div;
use crate::fraction_factors::{
    build_mul_from_factors_int_pow as build_mul_from_factors, collect_mul_factors_int_pow,
};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct DivAddSymmetricFactorRewrite {
    pub rewritten: ExprId,
}

/// Try to rewrite `Div(Add(...), Add(...))` by cancelling factors common to all
/// numerator terms and all denominator terms.
pub fn try_rewrite_div_add_symmetric_factor_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DivAddSymmetricFactorRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    if !matches!(ctx.get(num), Expr::Add(_, _)) || !matches!(ctx.get(den), Expr::Add(_, _)) {
        return None;
    }

    let mut num_terms = Vec::new();
    let mut den_terms = Vec::new();
    collect_add_terms(ctx, num, &mut num_terms);
    collect_add_terms(ctx, den, &mut den_terms);
    if num_terms.len() < 2 || den_terms.len() < 2 {
        return None;
    }

    let num_common = compute_common_factors(ctx, &num_terms);
    if num_common.is_empty() {
        return None;
    }
    let den_common = compute_common_factors(ctx, &den_terms);
    if den_common.is_empty() {
        return None;
    }

    let mut shared_common: Vec<(ExprId, i64)> = Vec::new();
    for (base, num_exp) in &num_common {
        if let Some(den_exp) = find_factor_exp(ctx, &den_common, *base) {
            let min_exp = (*num_exp).min(den_exp);
            if min_exp >= 1 {
                shared_common.push((*base, min_exp));
            }
        }
    }
    if shared_common.is_empty() {
        return None;
    }

    let new_num_terms = divide_terms_by_common(ctx, &num_terms, &shared_common);
    let new_den_terms = divide_terms_by_common(ctx, &den_terms, &shared_common);
    let new_num = rebuild_add(ctx, &new_num_terms);
    let new_den = rebuild_add(ctx, &new_den_terms);
    let rewritten = ctx.add(Expr::Div(new_num, new_den));
    if rewritten == expr {
        return None;
    }

    Some(DivAddSymmetricFactorRewrite { rewritten })
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

fn compute_common_factors(ctx: &Context, terms: &[ExprId]) -> Vec<(ExprId, i64)> {
    if terms.is_empty() {
        return Vec::new();
    }
    let mut common = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, terms[0]));
    for term_id in terms.iter().skip(1) {
        let term_map = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, *term_id));
        common.retain_mut(|(base, exp)| {
            if let Some(term_exp) = find_factor_exp(ctx, &term_map, *base) {
                *exp = (*exp).min(term_exp);
                *exp >= 1
            } else {
                false
            }
        });
        if common.is_empty() {
            return Vec::new();
        }
    }
    common
}

fn divide_terms_by_common(
    ctx: &mut Context,
    terms: &[ExprId],
    common: &[(ExprId, i64)],
) -> Vec<ExprId> {
    let mut new_terms = Vec::new();
    for term_id in terms {
        let (actual_term, is_negated) = match ctx.get(*term_id) {
            Expr::Neg(inner) => (*inner, true),
            _ => (*term_id, false),
        };
        let term_factors = collect_mul_factors_int_pow(ctx, actual_term);
        let mut quotient_factors: Vec<(ExprId, i64)> = Vec::new();
        for (base, exp) in term_factors {
            if matches!(ctx.get(base), Expr::Number(_)) {
                quotient_factors.push((base, exp));
                continue;
            }
            let common_exp = find_factor_exp(ctx, common, base).unwrap_or(0);
            let new_exp = exp - common_exp;
            if new_exp > 0 {
                quotient_factors.push((base, new_exp));
            }
        }
        let mut quotient = build_mul_from_factors(ctx, &quotient_factors);
        if is_negated {
            quotient = ctx.add(Expr::Neg(quotient));
        }
        new_terms.push(quotient);
    }
    new_terms
}

fn rebuild_add(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.len() == 1 {
        return terms[0];
    }
    let mut result = terms[0];
    for &term in terms.iter().skip(1) {
        result = ctx.add(Expr::Add(result, term));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_div_add_symmetric_factor_expr;
    use cas_ast::ordering::compare_expr;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn cancels_shared_common_factor_in_add_add_fraction() {
        let mut ctx = Context::new();
        let expr = parse("(x*a + x*b)/(x*c + x*d)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_div_add_symmetric_factor_expr(&mut ctx, expr).expect("rewrite");
        let (new_num, new_den) = if let Expr::Div(n, d) = ctx.get(rewrite.rewritten) {
            (*n, *d)
        } else {
            panic!("expected division");
        };
        let expected_num = parse("a+b", &mut ctx).expect("expected num");
        let expected_den = parse("c+d", &mut ctx).expect("expected den");
        assert_eq!(compare_expr(&ctx, new_num, expected_num), Ordering::Equal);
        assert_eq!(compare_expr(&ctx, new_den, expected_den), Ordering::Equal);
    }
}

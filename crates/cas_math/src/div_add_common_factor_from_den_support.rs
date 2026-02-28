//! Support for factoring common symbolic factors from additive numerators inside divisions.

use crate::build::mul2_raw;
use crate::expr_destructure::as_div;
use crate::fraction_factors::{
    build_mul_from_factors_int_pow as build_mul_from_factors, collect_mul_factors_int_pow,
};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct DivAddCommonFactorFromDenRewrite {
    pub rewritten: ExprId,
}

/// Try to factor the maximum common symbolic factor from an additive numerator
/// when those factors also appear in the denominator.
pub fn try_rewrite_div_add_common_factor_from_den_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DivAddCommonFactorFromDenRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    if !matches!(ctx.get(num), Expr::Add(_, _)) {
        return None;
    }

    let mut add_terms: Vec<ExprId> = Vec::new();
    collect_add_terms(ctx, num, &mut add_terms);
    if add_terms.len() < 2 {
        return None;
    }

    let den_map = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, den));
    if den_map.is_empty() {
        return None;
    }

    let mut common_map = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, add_terms[0]));

    for term_id in add_terms.iter().skip(1) {
        let term_map = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, *term_id));
        common_map.retain_mut(|(base, exp)| {
            if let Some(term_exp) = find_factor_exp(ctx, &term_map, *base) {
                *exp = (*exp).min(term_exp);
                *exp >= 1
            } else {
                false
            }
        });
        if common_map.is_empty() {
            return None;
        }
    }

    for (base, exp) in &mut common_map {
        if let Some(den_exp) = find_factor_exp(ctx, &den_map, *base) {
            *exp = (*exp).min(den_exp);
        } else {
            *exp = 0;
        }
    }
    common_map.retain(|(_, exp)| *exp >= 1);
    if common_map.is_empty() {
        return None;
    }

    let common_factors: Vec<(ExprId, i64)> = common_map.clone();
    let mut new_terms: Vec<ExprId> = Vec::new();

    for term_id in &add_terms {
        let (actual_term, is_neg) = match ctx.get(*term_id) {
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
            let common_exp = find_factor_exp(ctx, &common_map, base).unwrap_or(0);
            let new_exp = exp - common_exp;
            if new_exp > 0 {
                quotient_factors.push((base, new_exp));
            }
        }

        let quotient = build_mul_from_factors(ctx, &quotient_factors);
        let final_quotient = if is_neg {
            ctx.add(Expr::Neg(quotient))
        } else {
            quotient
        };
        new_terms.push(final_quotient);
    }

    let new_add = if new_terms.len() == 1 {
        new_terms[0]
    } else {
        let mut result = new_terms[0];
        for term in new_terms.iter().skip(1) {
            result = ctx.add(Expr::Add(result, *term));
        }
        result
    };

    let common_product = build_mul_from_factors(ctx, &common_factors);
    let new_num = mul2_raw(ctx, common_product, new_add);
    let rewritten = ctx.add(Expr::Div(new_num, den));
    if rewritten == expr {
        return None;
    }

    Some(DivAddCommonFactorFromDenRewrite { rewritten })
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

#[cfg(test)]
mod tests {
    use super::try_rewrite_div_add_common_factor_from_den_expr;
    use cas_ast::ordering::compare_expr;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn factors_common_symbolic_from_add_numerator() {
        let mut ctx = Context::new();
        let expr = parse("(x*a + x*b)/(x*c)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_div_add_common_factor_from_den_expr(&mut ctx, expr).expect("rewrite");
        let new_num = if let Expr::Div(new_num, _new_den) = ctx.get(rewrite.rewritten) {
            *new_num
        } else {
            panic!("expected division");
        };
        let expected_num = parse("x*(a+b)", &mut ctx).expect("parse expected");
        assert_eq!(compare_expr(&ctx, new_num, expected_num), Ordering::Equal);
    }
}

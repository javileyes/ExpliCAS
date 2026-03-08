//! Support for factoring common symbolic factors from additive denominators.

use crate::build::mul2_raw;
use crate::expr_destructure::as_div;
use crate::fraction_factors::{
    build_mul_from_factors_int_pow as build_mul_from_factors, collect_mul_factors_int_pow,
};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct DivDenFactorOutRewrite {
    pub rewritten: ExprId,
}

/// Try to rewrite `Div(num, Add(...))` by factoring common symbolic terms from
/// all additive denominator terms. A strict guard requires exact structural
/// overlap with at least one numerator factor to avoid factor/distribute loops.
pub fn try_rewrite_div_den_factor_out_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DivDenFactorOutRewrite> {
    let (num, den) = as_div(ctx, expr)?;
    if !matches!(ctx.get(den), Expr::Add(_, _)) {
        return None;
    }

    let mut den_terms = Vec::new();
    collect_add_terms(ctx, den, &mut den_terms);
    if den_terms.len() < 2 {
        return None;
    }

    let mut common_map = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, den_terms[0]));
    for &term_id in den_terms.iter().skip(1) {
        let term_map = factors_to_vec(ctx, &collect_mul_factors_int_pow(ctx, term_id));
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

    if common_map.is_empty() {
        return None;
    }

    // Strict guard: at least one exact structural match in numerator factors.
    let num_factors_raw = collect_mul_factors_int_pow(ctx, num);
    common_map.retain(|(den_base, _)| {
        num_factors_raw
            .iter()
            .any(|(num_base, _)| compare_expr(ctx, *den_base, *num_base) == Ordering::Equal)
    });
    if common_map.is_empty() {
        return None;
    }

    let common_factors: Vec<(ExprId, i64)> = common_map.clone();
    let mut new_den_terms: Vec<ExprId> = Vec::new();

    for &term_id in &den_terms {
        let (actual_term, is_neg) = match ctx.get(term_id) {
            Expr::Neg(inner) => (*inner, true),
            _ => (term_id, false),
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
        new_den_terms.push(final_quotient);
    }

    let common_product = build_mul_from_factors(ctx, &common_factors);
    let new_add = if new_den_terms.len() == 1 {
        new_den_terms[0]
    } else {
        let mut result = new_den_terms[0];
        for &term in new_den_terms.iter().skip(1) {
            result = ctx.add(Expr::Add(result, term));
        }
        result
    };

    let new_den = mul2_raw(ctx, common_product, new_add);
    let rewritten = ctx.add(Expr::Div(num, new_den));
    if rewritten == expr {
        return None;
    }

    Some(DivDenFactorOutRewrite { rewritten })
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
    use super::try_rewrite_div_den_factor_out_expr;
    use cas_ast::ordering::compare_expr;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn factors_common_symbolic_from_denominator_add() {
        let mut ctx = Context::new();
        let expr = parse("(x*a)/(x*b + x*c)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_div_den_factor_out_expr(&mut ctx, expr).expect("rewrite");
        let new_den = if let Expr::Div(_num, den) = ctx.get(rewrite.rewritten) {
            *den
        } else {
            panic!("expected division");
        };
        let expected = parse("x*(b+c)", &mut ctx).expect("expected");
        assert_eq!(compare_expr(&ctx, new_den, expected), Ordering::Equal);
    }
}

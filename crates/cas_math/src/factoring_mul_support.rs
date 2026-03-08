//! Structural support for extracting common multiplicative factors from sums.

use crate::expr_nary::{mul_factors, AddView, Sign};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use smallvec::SmallVec;
use std::cmp::Ordering;

/// Policy for n-ary common-multiplicative-factor extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractCommonMulFactorPolicy {
    pub max_terms: usize,
}

impl Default for ExtractCommonMulFactorPolicy {
    fn default() -> Self {
        Self { max_terms: 12 }
    }
}

/// Try to factor common multiplicative factors from an additive expression.
///
/// Pattern:
/// - `f*a + f*b + f*c -> f*(a+b+c)`
pub fn try_extract_common_mul_factor_expr(
    ctx: &mut Context,
    expr: ExprId,
    policy: ExtractCommonMulFactorPolicy,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    let terms = &view.terms;

    if terms.len() < 2 || terms.len() > policy.max_terms {
        return None;
    }

    let all_factors: SmallVec<[SmallVec<[ExprId; 8]>; 8]> = terms
        .iter()
        .map(|(term, _sign)| mul_factors(ctx, *term))
        .collect();

    let first_factors = &all_factors[0];
    let mut common_factors: SmallVec<[ExprId; 4]> = SmallVec::new();

    for &candidate in first_factors.iter() {
        if !is_extractable_factor(ctx, candidate) {
            continue;
        }

        if common_factors
            .iter()
            .any(|&cf| compare_expr(ctx, cf, candidate) == Ordering::Equal)
        {
            continue;
        }

        let in_all = all_factors
            .iter()
            .skip(1)
            .all(|factors| find_factor(ctx, factors, candidate).is_some());
        if in_all {
            common_factors.push(candidate);
        }
    }

    if common_factors.is_empty() {
        return None;
    }

    let mut quotient_terms: SmallVec<[(ExprId, Sign); 8]> = SmallVec::new();
    for (i, (_, sign)) in terms.iter().enumerate() {
        let mut remaining = all_factors[i].clone();
        for &cf in &common_factors {
            if let Some(idx) = find_factor(ctx, &remaining, cf) {
                remaining.remove(idx);
            } else {
                return None;
            }
        }

        let quotient = if remaining.is_empty() {
            ctx.num(1)
        } else {
            build_mul_chain(ctx, &remaining)
        };
        quotient_terms.push((quotient, *sign));
    }

    let inner_sum = AddView {
        root: expr, // placeholder for rebuild only
        terms: quotient_terms,
    }
    .rebuild(ctx);

    let common_product = build_mul_chain(ctx, &common_factors);
    let new_expr = ctx.add(Expr::Mul(common_product, inner_sum));

    let old_nodes = cas_ast::count_nodes(ctx, expr);
    let new_nodes = cas_ast::count_nodes(ctx, new_expr);
    if new_nodes > old_nodes {
        return None;
    }

    Some(new_expr)
}

fn find_factor(ctx: &Context, haystack: &[ExprId], needle: ExprId) -> Option<usize> {
    haystack
        .iter()
        .position(|&f| compare_expr(ctx, f, needle) == Ordering::Equal)
}

fn build_mul_chain(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    assert!(!factors.is_empty());
    let mut result = factors[0];
    for &f in &factors[1..] {
        result = ctx.add(Expr::Mul(result, f));
    }
    result
}

fn is_extractable_factor(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(_, _) => true,
        Expr::Constant(_) => true,
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                !n.is_integer()
            } else {
                true
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{try_extract_common_mul_factor_expr, ExtractCommonMulFactorPolicy};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn extracts_common_function_factor_from_sum() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)*a + sin(x)*b", &mut ctx).expect("parse");
        let out = try_extract_common_mul_factor_expr(
            &mut ctx,
            expr,
            ExtractCommonMulFactorPolicy::default(),
        );
        assert!(out.is_some());
    }

    #[test]
    fn rejects_when_no_extractable_common_factor() {
        let mut ctx = Context::new();
        let expr = parse("x*a + x*b", &mut ctx).expect("parse");
        let out = try_extract_common_mul_factor_expr(
            &mut ctx,
            expr,
            ExtractCommonMulFactorPolicy::default(),
        );
        assert!(out.is_none());
    }

    #[test]
    fn respects_term_count_policy() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)*a + sin(x)*b + sin(x)*c", &mut ctx).expect("parse");
        let out = try_extract_common_mul_factor_expr(
            &mut ctx,
            expr,
            ExtractCommonMulFactorPolicy { max_terms: 2 },
        );
        assert!(out.is_none());
    }
}

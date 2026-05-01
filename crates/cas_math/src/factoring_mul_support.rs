//! Structural support for extracting common multiplicative factors from sums.

use crate::expr_nary::{mul_factors, AddView, Sign};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;
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

    let common_integer_factor = common_factors
        .iter()
        .any(|&factor| is_compact_additive_integer_power(ctx, factor))
        .then(|| common_integer_factor(ctx, &all_factors))
        .flatten();

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
        if let Some(int_factor) = common_integer_factor {
            divide_integer_factor_from_factors(ctx, &mut remaining, int_factor)?;
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

    let mut extracted_factors = SmallVec::<[ExprId; 8]>::new();
    if let Some(int_factor) = common_integer_factor {
        extracted_factors.push(ctx.num(int_factor));
    }
    extracted_factors.extend(common_factors.iter().copied());

    let common_product = build_mul_chain(ctx, &extracted_factors);
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

fn common_integer_factor(ctx: &Context, all_factors: &[SmallVec<[ExprId; 8]>]) -> Option<i64> {
    let mut gcd = 0i64;
    for factors in all_factors {
        let coeff = single_integer_factor_value(ctx, factors)?.checked_abs()?;
        if coeff <= 1 {
            return None;
        }
        gcd = if gcd == 0 {
            coeff
        } else {
            gcd_abs_i64(gcd, coeff)
        };
        if gcd <= 1 {
            return None;
        }
    }

    (gcd > 1).then_some(gcd)
}

fn single_integer_factor_value(ctx: &Context, factors: &[ExprId]) -> Option<i64> {
    let mut value = None;
    for &factor in factors {
        let Some(candidate) = integer_number_value(ctx, factor) else {
            continue;
        };
        if candidate == 0 || value.is_some() {
            return None;
        }
        value = Some(candidate);
    }

    Some(value.unwrap_or(1))
}

fn divide_integer_factor_from_factors(
    ctx: &mut Context,
    factors: &mut SmallVec<[ExprId; 8]>,
    divisor: i64,
) -> Option<()> {
    for idx in 0..factors.len() {
        let Some(value) = integer_number_value(ctx, factors[idx]) else {
            continue;
        };
        if value % divisor != 0 {
            return None;
        }

        let quotient = value / divisor;
        if quotient == 1 {
            factors.remove(idx);
        } else {
            factors[idx] = ctx.num(quotient);
        }
        return Some(());
    }

    None
}

fn gcd_abs_i64(mut left: i64, mut right: i64) -> i64 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left.abs()
}

fn integer_number_value(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(n) if n.is_integer() => n.to_integer().to_i64(),
        _ => None,
    }
}

fn is_extractable_factor(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(_, _) => true,
        Expr::Constant(_) => true,
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                !n.is_integer() || is_compact_additive_integer_power(ctx, expr)
            } else {
                true
            }
        }
        _ => false,
    }
}

fn is_compact_additive_integer_power(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    let Some(power) = integer_number_value(ctx, *exp) else {
        return false;
    };
    if !(2..=8).contains(&power) {
        return false;
    }

    matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _))
        && additive_leaf_count_up_to(ctx, *base, 4).is_some()
        && cas_ast::count_nodes(ctx, *base) <= 25
}

fn additive_leaf_count_up_to(ctx: &Context, expr: ExprId, limit: usize) -> Option<usize> {
    if limit == 0 {
        return None;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            let left_count = additive_leaf_count_up_to(ctx, *left, limit)?;
            let remaining = limit.checked_sub(left_count)?;
            let right_count = additive_leaf_count_up_to(ctx, *right, remaining)?;
            let total = left_count + right_count;
            (total <= limit).then_some(total)
        }
        _ => Some(1),
    }
}

#[cfg(test)]
mod tests {
    use super::{try_extract_common_mul_factor_expr, ExtractCommonMulFactorPolicy};
    use cas_ast::{Context, Expr};
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
    fn extracts_common_compact_integer_power_and_scale_from_sum() {
        let mut ctx = Context::new();
        let expr = parse("6*(x^2+2*x+1)^2 + 6*x*(x^2+2*x+1)^2", &mut ctx).expect("parse");
        let out = try_extract_common_mul_factor_expr(
            &mut ctx,
            expr,
            ExtractCommonMulFactorPolicy::default(),
        )
        .expect("rewrite");

        let Expr::Mul(common, inner) = ctx.get(out) else {
            panic!("expected factored product, got {:?}", ctx.get(out));
        };
        assert!(
            matches!(ctx.get(*common), Expr::Mul(_, _)),
            "expected common product, got {:?}",
            ctx.get(*common)
        );
        assert!(
            matches!(ctx.get(*inner), Expr::Add(_, _)),
            "expected quotient sum, got {:?}",
            ctx.get(*inner)
        );
        assert!(cas_ast::count_nodes(&ctx, out) < cas_ast::count_nodes(&ctx, expr));
    }

    #[test]
    fn rejects_numeric_only_common_factor_without_structural_factor() {
        let mut ctx = Context::new();
        let expr = parse("2*x + 2*y", &mut ctx).expect("parse");
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

//! Derivative-cofactor recognizers for symbolic integration routes.
//!
//! This module owns the small mechanical operation of finding a unary
//! derivative-bearing factor inside a product and returning the remaining
//! multiplicative cofactor. Route-specific domain, scale, and primitive policy
//! stays in `symbolic_integration_support`.

use crate::expr_nary::{build_balanced_add, build_balanced_mul, mul_leaves, AddView, Sign};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;
use std::cmp::Ordering;

pub(crate) fn additive_cofactor_from_term_cofactors<F>(
    ctx: &mut Context,
    expr: ExprId,
    mut term_cofactor: F,
) -> Option<ExprId>
where
    F: FnMut(&mut Context, ExprId) -> Option<ExprId>,
{
    if let Some(cofactor) = term_cofactor(ctx, expr) {
        return Some(cofactor);
    }

    let add_view = AddView::from_expr(ctx, expr);
    if add_view.terms.len() < 2 {
        return None;
    }

    let mut cofactor_terms = Vec::with_capacity(add_view.terms.len());
    for (term, sign) in add_view.terms {
        let cofactor = term_cofactor(ctx, term)?;
        cofactor_terms.push(match sign {
            Sign::Pos => cofactor,
            Sign::Neg => ctx.add(Expr::Neg(cofactor)),
        });
    }

    Some(build_balanced_add(ctx, &cofactor_terms))
}

pub(crate) fn product_cofactor_excluding_unary_builtin_arg<F>(
    ctx: &mut Context,
    term: ExprId,
    builtin: BuiltinFn,
    arg_matches: F,
) -> Option<ExprId>
where
    F: FnMut(&mut Context, ExprId) -> bool,
{
    product_cofactor_excluding_unary_builtin_arg_with_multiplicity(
        ctx,
        term,
        builtin,
        arg_matches,
        false,
    )
}

pub(crate) fn unique_product_cofactor_excluding_unary_builtin_arg<F>(
    ctx: &mut Context,
    term: ExprId,
    builtin: BuiltinFn,
    arg_matches: F,
) -> Option<ExprId>
where
    F: FnMut(&mut Context, ExprId) -> bool,
{
    product_cofactor_excluding_unary_builtin_arg_with_multiplicity(
        ctx,
        term,
        builtin,
        arg_matches,
        true,
    )
}

pub(crate) fn factors_excluding_index(factors: &[ExprId], excluded_index: usize) -> Vec<ExprId> {
    factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != excluded_index).then_some(*factor))
        .collect()
}

pub(crate) fn factors_excluding_two_indices(
    factors: &[ExprId],
    first_excluded_index: usize,
    second_excluded_index: usize,
) -> Vec<ExprId> {
    factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| {
            (idx != first_excluded_index && idx != second_excluded_index).then_some(*factor)
        })
        .collect()
}

pub(crate) fn factor_product_or_one(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    if factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, factors)
    }
}

pub(crate) fn factor_product_excluding_index(
    ctx: &mut Context,
    factors: &[ExprId],
    excluded_index: usize,
) -> ExprId {
    let cofactor_factors = factors_excluding_index(factors, excluded_index);
    factor_product_or_one(ctx, &cofactor_factors)
}

pub(crate) fn signed_factor_product_excluding_index(
    ctx: &mut Context,
    factors: &[ExprId],
    excluded_index: usize,
    sign: Sign,
) -> ExprId {
    let cofactor = factor_product_excluding_index(ctx, factors, excluded_index);
    match sign {
        Sign::Pos => cofactor,
        Sign::Neg => ctx.add(Expr::Neg(cofactor)),
    }
}

pub(crate) fn indexed_signed_matching_unary_factor(
    ctx: &Context,
    factors: &[ExprId],
    builtin: BuiltinFn,
    expected_arg: ExprId,
) -> Option<(usize, BigRational)> {
    factors.iter().enumerate().find_map(|(idx, factor)| {
        let (arg, sign) = signed_unary_builtin_arg(ctx, *factor, builtin)?;
        (compare_expr(ctx, arg, expected_arg) == Ordering::Equal).then_some((idx, sign))
    })
}

fn product_cofactor_excluding_unary_builtin_arg_with_multiplicity<F>(
    ctx: &mut Context,
    term: ExprId,
    builtin: BuiltinFn,
    mut arg_matches: F,
    require_unique_match: bool,
) -> Option<ExprId>
where
    F: FnMut(&mut Context, ExprId) -> bool,
{
    let factors = mul_leaves(ctx, term);
    let mut matching_index = None;

    for (idx, factor) in factors.iter().enumerate() {
        let Some(candidate_arg) = unary_builtin_arg(ctx, *factor, builtin) else {
            continue;
        };
        if arg_matches(ctx, candidate_arg) {
            if require_unique_match && matching_index.is_some() {
                return None;
            }
            matching_index = Some(idx);
            if !require_unique_match {
                break;
            }
        }
    }

    let matching_index = matching_index?;
    Some(factor_product_excluding_index(
        ctx,
        &factors,
        matching_index,
    ))
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn signed_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            unary_builtin_arg(ctx, *inner, builtin).map(|arg| (arg, -BigRational::one()))
        }
        _ => unary_builtin_arg(ctx, expr, builtin).map(|arg| (arg, BigRational::one())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::ordering::compare_expr;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use std::cmp::Ordering;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn returns_product_cofactor_after_matching_unary_builtin_arg() {
        let mut ctx = Context::new();
        let term = parse("2*x*sinh(x^2)", &mut ctx).unwrap();
        let expected_arg = parse("x^2", &mut ctx).unwrap();

        let cofactor = product_cofactor_excluding_unary_builtin_arg(
            &mut ctx,
            term,
            BuiltinFn::Sinh,
            |ctx, candidate_arg| compare_expr(ctx, candidate_arg, expected_arg) == Ordering::Equal,
        )
        .unwrap();

        assert_eq!(rendered(&ctx, cofactor), "2 * x");
    }

    #[test]
    fn returns_one_when_matched_factor_is_the_whole_term() {
        let mut ctx = Context::new();
        let term = parse("cosh(x)", &mut ctx).unwrap();
        let expected_arg = parse("x", &mut ctx).unwrap();

        let cofactor = product_cofactor_excluding_unary_builtin_arg(
            &mut ctx,
            term,
            BuiltinFn::Cosh,
            |ctx, candidate_arg| compare_expr(ctx, candidate_arg, expected_arg) == Ordering::Equal,
        )
        .unwrap();

        assert_eq!(rendered(&ctx, cofactor), "1");
    }

    #[test]
    fn rejects_when_unary_argument_does_not_match() {
        let mut ctx = Context::new();
        let term = parse("2*x*sinh(x^2)", &mut ctx).unwrap();
        let expected_arg = parse("x", &mut ctx).unwrap();

        assert!(product_cofactor_excluding_unary_builtin_arg(
            &mut ctx,
            term,
            BuiltinFn::Sinh,
            |ctx, candidate_arg| compare_expr(ctx, candidate_arg, expected_arg) == Ordering::Equal,
        )
        .is_none());
    }

    #[test]
    fn unique_product_cofactor_rejects_multiple_matching_unary_factors() {
        let mut ctx = Context::new();
        let term = parse("2*x*sinh(x^2)*sinh(x^2)", &mut ctx).unwrap();
        let expected_arg = parse("x^2", &mut ctx).unwrap();

        assert!(unique_product_cofactor_excluding_unary_builtin_arg(
            &mut ctx,
            term,
            BuiltinFn::Sinh,
            |ctx, candidate_arg| compare_expr(ctx, candidate_arg, expected_arg) == Ordering::Equal,
        )
        .is_none());
    }

    #[test]
    fn rebuilds_additive_cofactor_from_signed_term_cofactors() {
        let mut ctx = Context::new();
        let expr = parse("2*x*sinh(x^2) - 3*x*sinh(x^2)", &mut ctx).unwrap();
        let expected_arg = parse("x^2", &mut ctx).unwrap();

        let cofactor = additive_cofactor_from_term_cofactors(&mut ctx, expr, |ctx, term| {
            product_cofactor_excluding_unary_builtin_arg(ctx, term, BuiltinFn::Sinh, |ctx, arg| {
                compare_expr(ctx, arg, expected_arg) == Ordering::Equal
            })
        })
        .unwrap();

        assert_eq!(rendered(&ctx, cofactor), "2 * x - 3 * x");
    }

    #[test]
    fn rejects_additive_cofactor_when_any_term_does_not_match() {
        let mut ctx = Context::new();
        let expr = parse("2*x*sinh(x^2) + 1", &mut ctx).unwrap();
        let expected_arg = parse("x^2", &mut ctx).unwrap();

        assert!(
            additive_cofactor_from_term_cofactors(&mut ctx, expr, |ctx, term| {
                product_cofactor_excluding_unary_builtin_arg(
                    ctx,
                    term,
                    BuiltinFn::Sinh,
                    |ctx, arg| compare_expr(ctx, arg, expected_arg) == Ordering::Equal,
                )
            })
            .is_none()
        );
    }
}

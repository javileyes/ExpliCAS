//! Shared sqrt-chain coefficient recognizers for calculus presentation routes.
//!
//! These helpers are intentionally narrower than general simplification. They
//! only recover rational coefficients around known `sqrt(radicand)` factors for
//! derivative/integrand presentation paths that already own the surrounding
//! domain and route policy.

use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_utils::calculus_sqrt_like_radicand;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn sqrt_chain_linear_derivative_coeff(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative = radicand_poly.derivative();
    if derivative.is_zero() || derivative.degree() != 0 {
        return None;
    }
    let derivative_expr = derivative.to_expr(ctx);
    let derivative_coeff = cas_ast::views::as_rational_const(ctx, derivative_expr, 8)?;
    Some(derivative_coeff / BigRational::from_integer(2.into()))
}

pub(super) fn sqrt_chain_factor_coeff_over_sqrt(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    radicand: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if let Some((idx, _)) = numerator_factors
        .iter()
        .enumerate()
        .find(|(_, factor)| reciprocal_sqrt_factor_matches(ctx, **factor, radicand))
    {
        let numerator_rest: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        return rational_factor_quotient(ctx, &numerator_rest, denominator_factors);
    }

    if let Some((idx, _)) = denominator_factors
        .iter()
        .enumerate()
        .find(|(_, factor)| sqrt_factor_matches(ctx, **factor, radicand))
    {
        let denominator_rest: Vec<_> = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        return rational_factor_quotient(ctx, numerator_factors, &denominator_rest);
    }

    let (sqrt_idx, _) = numerator_factors
        .iter()
        .enumerate()
        .find(|(_, factor)| sqrt_factor_matches(ctx, **factor, radicand))?;
    let (radicand_idx, denominator_radicand_scale) = denominator_factors
        .iter()
        .enumerate()
        .find_map(|(idx, factor)| {
            denominator_factor_radicand_scale(ctx, *factor, radicand, var_name)
                .map(|scale| (idx, scale))
        })?;
    let numerator_rest: Vec<_> = numerator_factors
        .iter()
        .enumerate()
        .filter_map(|(factor_idx, factor)| (factor_idx != sqrt_idx).then_some(*factor))
        .collect();
    let denominator_rest: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(factor_idx, factor)| (factor_idx != radicand_idx).then_some(*factor))
        .collect();
    rational_factor_quotient(ctx, &numerator_rest, &denominator_rest)
        .map(|coeff| coeff / denominator_radicand_scale)
}

fn sqrt_factor_matches(ctx: &mut Context, factor: ExprId, radicand: ExprId) -> bool {
    calculus_sqrt_like_radicand(ctx, factor)
        .is_some_and(|base| compare_expr(ctx, base, radicand) == std::cmp::Ordering::Equal)
}

fn denominator_factor_radicand_scale(
    ctx: &Context,
    factor: ExprId,
    radicand: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if compare_expr(ctx, factor, radicand) == std::cmp::Ordering::Equal {
        return Some(BigRational::one());
    }

    let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    polynomial_scale(&factor_poly, &radicand_poly)
}

fn polynomial_scale(factor: &Polynomial, base: &Polynomial) -> Option<BigRational> {
    if base.is_zero() {
        return None;
    }

    let len = factor.coeffs.len().max(base.coeffs.len());
    let mut scale = None;
    for idx in 0..len {
        let factor_coeff = factor
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let base_coeff = base
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if base_coeff.is_zero() {
            if !factor_coeff.is_zero() {
                return None;
            }
            continue;
        }

        let candidate = factor_coeff / base_coeff;
        if candidate.is_zero() {
            return None;
        }
        match &scale {
            Some(existing) if *existing != candidate => return None,
            Some(_) => {}
            None => scale = Some(candidate),
        }
    }

    scale
}

fn reciprocal_sqrt_factor_matches(ctx: &mut Context, factor: ExprId, radicand: ExprId) -> bool {
    matches!(
        ctx.get(factor),
        Expr::Pow(base, exp)
            if compare_expr(ctx, *base, radicand) == std::cmp::Ordering::Equal
                && cas_ast::views::as_rational_const(ctx, *exp, 8)
                    == Some(BigRational::new((-1).into(), 2.into()))
    )
}

fn rational_factor_quotient(
    ctx: &Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
) -> Option<BigRational> {
    let numerator = rational_factor_product(ctx, numerator_factors)?;
    let denominator = rational_factor_product(ctx, denominator_factors)?;
    (!denominator.is_zero()).then_some(numerator / denominator)
}

fn rational_factor_product(ctx: &Context, factors: &[ExprId]) -> Option<BigRational> {
    factors.iter().try_fold(BigRational::one(), |acc, factor| {
        cas_ast::views::as_rational_const(ctx, *factor, 8).map(|value| acc * value)
    })
}

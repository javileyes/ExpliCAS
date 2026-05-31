//! Compact derivative presentation for polynomial-over-square-root routes.
//!
//! This module owns the `p(x) / sqrt(q(x))` presentation family and its
//! positive-radicand domain wrapper. Nearby log/root quotient families have a
//! separate owner because their parameter-scale policy is narrower.

use super::domain_checks::positive_polynomial_radicand_required_conditions;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::calculus_sqrt_like_radicand;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::One;

pub(super) fn polynomial_over_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let radicand = calculus_sqrt_like_radicand(ctx, denominator_expr)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    polynomial_over_sqrt_polynomial_derivative_presentation_from_parts(
        ctx,
        numerator_expr,
        radicand,
        &radicand_poly,
        var_name,
    )
}

fn polynomial_over_sqrt_polynomial_derivative_presentation_from_parts(
    ctx: &mut Context,
    numerator_expr: ExprId,
    radicand: ExprId,
    radicand_poly: &Polynomial,
    var_name: &str,
) -> Option<ExprId> {
    let numerator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, numerator_expr, var_name)?;
    if radicand_poly.is_zero() {
        return None;
    }
    let numerator_derivative = numerator_poly.derivative();
    if numerator_derivative.is_zero() {
        return None;
    }
    if numerator_derivative.degree() > 2 {
        return None;
    }

    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let mut numerator_result_poly = numerator_derivative
        .mul(radicand_poly)
        .mul(&two_poly)
        .sub(&numerator_poly.mul(&radicand_poly.derivative()));
    if numerator_result_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let mut cancel_radicand_from_denominator = false;
    if let Ok((quotient, remainder)) = numerator_result_poly.div_rem(radicand_poly) {
        if remainder.is_zero() {
            numerator_result_poly = quotient;
            cancel_radicand_from_denominator = true;
        }
    }
    let mut lift_radicand_to_numerator = false;
    if cancel_radicand_from_denominator {
        if let Ok((quotient, remainder)) = numerator_result_poly.div_rem(radicand_poly) {
            if remainder.is_zero() {
                numerator_result_poly = quotient;
                lift_radicand_to_numerator = true;
            }
        }
    }

    let raw_numerator = numerator_result_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let coefficient = numerator_content / BigRational::new(2.into(), 1.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let numerator_core = if lift_radicand_to_numerator {
        if cas_ast::views::as_rational_const(ctx, numerator_core, 8)
            .is_some_and(|value| value.is_one())
        {
            sqrt_radicand
        } else {
            cas_math::expr_nary::build_balanced_mul(ctx, &[numerator_core, sqrt_radicand])
        }
    } else {
        numerator_core
    };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);
    let core_denominator = if cancel_radicand_from_denominator {
        if lift_radicand_to_numerator {
            ctx.num(1)
        } else {
            sqrt_radicand
        }
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, sqrt_radicand])
    };
    let denominator_coeff_is_one = denominator_coeff == BigRational::one();
    let denominator = if denominator_coeff_is_one {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    if lift_radicand_to_numerator && denominator_coeff_is_one {
        return Some(numerator);
    }

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(crate) fn polynomial_over_sqrt_polynomial_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
        return None;
    };
    let radicand = calculus_sqrt_like_radicand(ctx, denominator_expr)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result = polynomial_over_sqrt_polynomial_derivative_presentation_from_parts(
        ctx,
        numerator_expr,
        radicand,
        &radicand_poly,
        var_name,
    )?;
    let required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, &radicand_poly);
    Some((result, required_conditions))
}

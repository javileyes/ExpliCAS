use super::differentiation::differentiate;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation, strictly_positive_quadratic_on_reals,
};
use super::positive_quadratic_square_result_presentation::compact_positive_quadratic_square_derivative_result;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};
use cas_ast::{Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn rational_over_matching_denominator_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    denominator_poly: &Polynomial,
    var_name: &str,
) -> Option<BigRational> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };
    let numerator_value = cas_ast::views::as_rational_const(ctx, numerator, 8)?;
    let (observed_denominator_core, observed_denominator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator);
    if observed_denominator_content.is_zero() {
        return None;
    }
    let observed_denominator =
        polynomial_radicand_for_calculus_presentation(ctx, observed_denominator_core, var_name)?;
    (observed_denominator == *denominator_poly)
        .then_some(numerator_value / observed_denominator_content)
}

pub(super) fn positive_quadratic_quotient_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(target).clone() else {
        return None;
    };
    let (mut denominator_core, mut denominator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator);
    if denominator_content.is_zero() {
        return None;
    }
    let mut denominator_core_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_core, var_name)?;
    if !strictly_positive_quadratic_on_reals(&denominator_core_poly) {
        let negated_denominator_core_poly = Polynomial::new(
            denominator_core_poly
                .coeffs
                .iter()
                .map(|coeff| -coeff.clone())
                .collect(),
            denominator_core_poly.var.clone(),
        );
        if !strictly_positive_quadratic_on_reals(&negated_denominator_core_poly) {
            return None;
        }
        denominator_core = negated_denominator_core_poly.to_expr(ctx);
        denominator_content = -denominator_content;
        denominator_core_poly = negated_denominator_core_poly;
    }

    let numerator_derivative = differentiate(ctx, numerator, var_name)?;
    let numerator_derivative_scale = rational_over_matching_denominator_for_calculus_presentation(
        ctx,
        numerator_derivative,
        &denominator_core_poly,
        var_name,
    )?;
    if numerator_derivative_scale.is_zero() {
        return None;
    }

    let denominator_derivative = denominator_core_poly.derivative().to_expr(ctx);
    let (denominator_derivative_core, denominator_derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, denominator_derivative);
    let reciprocal_content = BigRational::one() / denominator_content;
    let scaled_numerator_derivative = rational_const_for_calculus_presentation(
        ctx,
        numerator_derivative_scale * reciprocal_content.clone(),
    );
    let scaled_denominator_derivative_coeff = denominator_derivative_content * reciprocal_content;
    let compact_numerator = if scaled_denominator_derivative_coeff.is_negative() {
        let scaled_denominator_derivative = scale_expr_for_calculus_presentation(
            ctx,
            -scaled_denominator_derivative_coeff,
            denominator_derivative_core,
        );
        let quotient_term = cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[scaled_denominator_derivative, numerator],
        );
        ctx.add(Expr::Add(scaled_numerator_derivative, quotient_term))
    } else {
        let scaled_denominator_derivative = scale_expr_for_calculus_presentation(
            ctx,
            scaled_denominator_derivative_coeff,
            denominator_derivative_core,
        );
        let quotient_term = cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[scaled_denominator_derivative, numerator],
        );
        ctx.add(Expr::Sub(scaled_numerator_derivative, quotient_term))
    };
    let two = ctx.num(2);
    let compact_denominator = ctx.add(Expr::Pow(denominator_core, two));
    let compact = ctx.add(Expr::Div(compact_numerator, compact_denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn positive_quadratic_square_derivative_result_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let result = differentiate(ctx, target, var_name)?;
    compact_positive_quadratic_square_derivative_result(ctx, result, var_name)
}

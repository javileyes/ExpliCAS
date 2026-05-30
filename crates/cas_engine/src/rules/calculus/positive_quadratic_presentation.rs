use super::differentiation::differentiate;
use super::polynomial_support::{
    polynomial_derivative_expr_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation, square_of_strictly_positive_quadratic_arg,
    strictly_positive_quadratic_on_reals,
};
use super::power_result_presentation::compact_positive_quadratic_square_derivative_result;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_bigint::BigInt;
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

fn positive_rational_denominator_scaled_base(
    ctx: &mut Context,
    base: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational, ExprId)> {
    if let Expr::Div(numerator, denominator) = ctx.get(base) {
        let scale = cas_ast::views::as_rational_const(ctx, *denominator, 8)?;
        if !scale.is_positive() {
            return None;
        }

        let numerator_poly =
            polynomial_radicand_for_calculus_presentation(ctx, *numerator, var_name)?;
        return strictly_positive_quadratic_on_reals(&numerator_poly)
            .then_some((*numerator, scale, base));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, base);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut non_rational_factor = None;
    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            if !value.is_positive() {
                return None;
            }
            scale *= value;
            continue;
        }

        if non_rational_factor.replace(factor).is_some() {
            return None;
        }
    }

    if scale.is_one() {
        return None;
    }
    let core = non_rational_factor?;
    let core_poly = polynomial_radicand_for_calculus_presentation(ctx, core, var_name)?;
    if !strictly_positive_quadratic_on_reals(&core_poly) {
        return None;
    }

    let compact_scaled_base = if scale.numer() == &BigInt::one() {
        let denominator = rational_const_for_calculus_presentation(
            ctx,
            BigRational::from_integer(scale.denom().clone()),
        );
        ctx.add(Expr::Div(core, denominator))
    } else {
        scale_expr_for_calculus_presentation(ctx, scale.clone(), core)
    };

    Some((core, BigRational::one() / scale, compact_scaled_base))
}

pub(super) fn inverse_reciprocal_trig_positive_quadratic_square_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let base = square_of_strictly_positive_quadratic_arg(ctx, args[0], var_name)?;
    let derivative = polynomial_derivative_expr_for_calculus_presentation(ctx, base, var_name)?;
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let (denominator_base, numerator_scale, gap_base) = positive_rational_denominator_scaled_base(
        ctx, base, var_name,
    )
    .unwrap_or((base, BigRational::one(), base));
    let numerator = signed_numerator_for_calculus_presentation(
        ctx,
        sign * derivative_content * BigRational::from_integer(2.into()) * numerator_scale,
        derivative_core,
    );

    let four = ctx.num(4);
    let base_fourth = ctx.add(Expr::Pow(gap_base, four));
    let one = ctx.num(1);
    let gap = ctx.add(Expr::Sub(base_fourth, one));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_base, sqrt_gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

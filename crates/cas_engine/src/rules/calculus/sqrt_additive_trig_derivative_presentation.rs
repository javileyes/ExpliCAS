use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Zero};

use super::differentiation::differentiate;
use super::elementary_variable_term_presentation::{
    scaled_exp_variable_term_for_calculus_presentation,
    scaled_ln_variable_arg_for_calculus_presentation,
    scaled_reciprocal_variable_term_for_calculus_presentation,
    scaled_sqrt_variable_term_for_calculus_presentation,
};
use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_compaction::{
    bounded_sin_cos_term_bound_for_calculus_presentation,
    compact_double_angle_sine_products_for_calculus_presentation,
    compact_numeric_mul_factors_for_calculus_presentation,
    compact_small_power_exponents_for_calculus_presentation,
    distribute_half_over_additive_numerator_for_calculus_presentation,
    signed_add_terms_for_calculus_presentation,
};
use super::presentation_utils::sqrt_raw_for_calculus_presentation;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation, split_numeric_scale_single_core,
};

pub(crate) fn sqrt_additive_trig_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let presentation_radicand =
        compact_double_angle_sine_products_for_calculus_presentation(ctx, radicand)
            .or_else(|| signed_add_terms_for_calculus_presentation(ctx, radicand))
            .unwrap_or(radicand);
    let derivative_parts = additive_trig_polynomial_sqrt_radicand_derivative_for_presentation(
        ctx,
        presentation_radicand,
        var_name,
    )?;
    let required_conditions = derivative_parts.required_conditions;
    if let Some(derivative_denominator) = derivative_parts.denominator {
        let numerator =
            compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative_parts.numerator);
        let two =
            rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
        let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, presentation_radicand);
        let denominator = cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[two, derivative_denominator, sqrt_radicand],
        );
        let compact = ctx.add_raw(Expr::Div(numerator, denominator));
        return Some((
            cas_ast::hold::wrap_hold(ctx, compact),
            radicand,
            required_conditions,
        ));
    }

    let derivative = derivative_parts.numerator;
    let derivative = compact_small_power_exponents_for_calculus_presentation(ctx, derivative);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some((ctx.num(0), radicand, required_conditions));
    }

    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let distributed_numerator = if coefficient == BigRational::new(1.into(), 2.into()) {
        distribute_half_over_additive_numerator_for_calculus_presentation(ctx, derivative_core)
    } else {
        None
    };
    let (numerator, denominator_coeff) = if let Some(numerator) = distributed_numerator {
        (numerator, BigRational::one())
    } else {
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        (
            scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core),
            denominator_coeff,
        )
    };
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, presentation_radicand);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        radicand,
        required_conditions,
    ))
}

struct AdditiveTrigPolynomialDerivativeForPresentation {
    numerator: ExprId,
    denominator: Option<ExprId>,
    required_conditions: Vec<crate::ImplicitCondition>,
}

fn additive_trig_polynomial_sqrt_radicand_derivative_for_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<AdditiveTrigPolynomialDerivativeForPresentation> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut has_trig_term = false;
    let mut has_variable_dependency = false;
    let mut derivative_terms = Vec::new();
    let mut denominator = None;
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            has_trig_term = true;
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                derivative_terms.push(derivative);
            }
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            derivative_terms.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some((ln_scale, ln_arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if denominator.is_some_and(|existing| existing != ln_arg) {
                return None;
            }
            denominator = Some(ln_arg);
            derivative_terms.push(rational_const_for_calculus_presentation(ctx, ln_scale));
            required_conditions.push(crate::ImplicitCondition::Positive(ln_arg));
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            let neg_half = ctx.rational(-1, 2);
            let reciprocal_sqrt = ctx.add_raw(Expr::Pow(sqrt_arg, neg_half));
            let derivative = scale_expr_for_calculus_presentation(
                ctx,
                sqrt_scale * BigRational::new(1.into(), 2.into()),
                reciprocal_sqrt,
            );
            derivative_terms.push(derivative);
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if let Some((reciprocal_scale, reciprocal_arg)) =
            scaled_reciprocal_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            let two = ctx.num(2);
            let reciprocal_denominator = ctx.add(Expr::Pow(reciprocal_arg, two));
            if denominator.is_some_and(|existing| existing != reciprocal_denominator) {
                return None;
            }
            denominator = Some(reciprocal_denominator);
            derivative_terms.push(rational_const_for_calculus_presentation(
                ctx,
                -reciprocal_scale,
            ));
            required_conditions.push(crate::ImplicitCondition::NonZero(reciprocal_arg));
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            derivative_terms.push(derivative.to_expr(ctx));
        }
    }

    if !has_trig_term || !has_variable_dependency {
        return None;
    }

    let numerator = if let Some(denominator) = denominator {
        let scaled_terms: Vec<_> = derivative_terms
            .into_iter()
            .map(|term| {
                if cas_ast::views::as_rational_const(ctx, term, 8).is_some() {
                    term
                } else {
                    ctx.add(Expr::Mul(denominator, term))
                }
            })
            .collect();
        if scaled_terms.is_empty() {
            ctx.num(0)
        } else {
            cas_math::expr_nary::build_balanced_add(ctx, &scaled_terms)
        }
    } else if derivative_terms.is_empty() {
        ctx.num(0)
    } else {
        cas_math::expr_nary::build_balanced_add(ctx, &derivative_terms)
    };
    Some(AdditiveTrigPolynomialDerivativeForPresentation {
        numerator,
        denominator,
        required_conditions,
    })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::sqrt_additive_trig_polynomial_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn additive_trig_reciprocal_subtraction_sqrt_derivative_presentation_is_direct() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(sin(2*x)+cos(x)-2/x)", &mut ctx).unwrap();
        let (compact, required_positive, required_conditions) =
            sqrt_additive_trig_polynomial_derivative_presentation(&mut ctx, expr, "x")
                .unwrap_or_else(|| panic!("subtracted reciprocal trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(2 * cos(2 * x) * x^2 + 2 - sin(x) * x^2) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) - 2 / x))"
        );
        assert_eq!(
            rendered(&ctx, required_positive),
            "sin(2 * x) + cos(x) - 2 / x"
        );
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn additive_trig_reciprocal_addition_sqrt_derivative_presentation_stays_direct() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(sin(2*x)+cos(x)+1/x)", &mut ctx).unwrap();
        let (compact, required_positive, required_conditions) =
            sqrt_additive_trig_polynomial_derivative_presentation(&mut ctx, expr, "x")
                .unwrap_or_else(|| panic!("added reciprocal trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(2 * cos(2 * x) * x^2 - sin(x) * x^2 - 1) / (2 * x^2 * sqrt(sin(2 * x) + cos(x) + 1 / x))"
        );
        assert_eq!(
            rendered(&ctx, required_positive),
            "sin(2 * x) + cos(x) + 1 / x"
        );
        assert_eq!(required_conditions.len(), 1);
    }
}

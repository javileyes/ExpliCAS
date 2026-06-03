use super::derivative_result_scaling_presentation::scale_compact_derivative_by_rational;
use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::domain_checks::positive_polynomial_radicand_and_nonzero_required_conditions;
use super::polynomial_power_presentation::{
    polynomial_power_for_calculus_presentation,
    positive_integer_polynomial_power_for_calculus_presentation,
};
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::{squared_expr, unwrap_internal_hold_for_calculus};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    rational_scaled_single_factor_allow_unit, scale_expr_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use super::shifted_sqrt_derivative_presentation::inverse_tangent_reciprocal_sqrt_shifted_sqrt_product_derivative_presentation;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

fn nonnegative_integer_offset_from_half_exponent(value: BigRational) -> Option<usize> {
    let offset = value - BigRational::new(1.into(), 2.into());
    if offset.is_negative() || !offset.is_integer() {
        return None;
    }
    offset.to_integer().to_usize()
}

fn sqrt_times_polynomial_factor_parts(
    ctx: &Context,
    factor: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
    if let Some(radicand) = extract_square_root_base(ctx, factor) {
        polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
        return Some((radicand, Polynomial::one(var_name.to_string())));
    }

    let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
    let offset = nonnegative_integer_offset_from_half_exponent(exponent)?;
    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
    Some((
        base,
        polynomial_power_for_calculus_presentation(&base_poly, offset),
    ))
}

fn denominator_product_sqrt_polynomial_parts(
    ctx: &Context,
    denominator_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    let mut radicand = None;
    let mut polynomial = Polynomial::one(var_name.to_string());
    for factor in cas_math::expr_nary::mul_leaves(ctx, denominator_expr) {
        if let Some((factor_radicand, factor_polynomial)) =
            sqrt_times_polynomial_factor_parts(ctx, factor, var_name)
        {
            if factor_polynomial != Polynomial::one(var_name.to_string()) {
                return None;
            }
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
        } else {
            let factor_poly = polynomial_radicand_for_calculus_presentation(ctx, factor, var_name)?;
            polynomial = polynomial.mul(&factor_poly);
        }
    }

    Some((radicand?, polynomial))
}

fn denominator_sum_sqrt_polynomial_parts(
    ctx: &Context,
    denominator_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    let mut radicand = None;
    let mut polynomial = Polynomial::zero(var_name.to_string());
    let terms = cas_math::expr_nary::AddView::from_expr(ctx, denominator_expr).terms;
    if terms.len() < 2 {
        return None;
    }
    for (term, sign) in terms {
        let mut term_radicand = None;
        let mut term_polynomial = Polynomial::one(var_name.to_string());
        for factor in cas_math::expr_nary::mul_leaves(ctx, term) {
            if let Some((factor_radicand, factor_polynomial)) =
                sqrt_times_polynomial_factor_parts(ctx, factor, var_name)
            {
                if term_radicand.replace(factor_radicand).is_some() {
                    return None;
                }
                term_polynomial = term_polynomial.mul(&factor_polynomial);
            } else {
                let factor_poly =
                    polynomial_radicand_for_calculus_presentation(ctx, factor, var_name)?;
                term_polynomial = term_polynomial.mul(&factor_poly);
            }
        }
        let term_radicand = term_radicand?;
        if let Some(existing) = radicand {
            if compare_expr(ctx, existing, term_radicand) != std::cmp::Ordering::Equal {
                return None;
            }
        } else {
            radicand = Some(term_radicand);
        }
        polynomial = match sign {
            cas_math::expr_nary::Sign::Pos => polynomial.add(&term_polynomial),
            cas_math::expr_nary::Sign::Neg => polynomial.sub(&term_polynomial),
        };
    }

    Some((radicand?, polynomial))
}

fn denominator_sqrt_polynomial_parts(
    ctx: &Context,
    denominator_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, Polynomial)> {
    denominator_product_sqrt_polynomial_parts(ctx, denominator_expr, var_name)
        .or_else(|| denominator_sum_sqrt_polynomial_parts(ctx, denominator_expr, var_name))
}

fn negative_half_power_target_parts(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, Polynomial)> {
    let target = cas_ast::hold::unwrap_internal_hold(ctx, target);
    let mut scale = BigRational::one();
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
            return None;
        };
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent != BigRational::new((-1).into(), 2.into()) {
            return None;
        }
        polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
        if radicand.replace(base).is_some() {
            return None;
        }
    }

    Some((scale, radicand?, Polynomial::one(var_name.to_string())))
}

pub(super) fn reciprocal_sqrt_polynomial_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (numerator_scale, radicand, denominator_poly, denominator_power_factor) =
        if let Some((scale, radicand, denominator_poly)) =
            negative_half_power_target_parts(ctx, target, var_name)
        {
            (scale, radicand, denominator_poly, None)
        } else {
            let Expr::Div(numerator_expr, denominator_expr) = ctx.get(target).clone() else {
                return None;
            };
            let numerator_scale = cas_ast::views::as_rational_const(ctx, numerator_expr, 8)?;
            let (radicand, denominator_poly) =
                denominator_sqrt_polynomial_parts(ctx, denominator_expr, var_name)?;
            let denominator_power_factor =
                sqrt_denominator_positive_integer_power_factor(ctx, denominator_expr, var_name);
            (
                numerator_scale,
                radicand,
                denominator_poly,
                denominator_power_factor,
            )
        };
    if numerator_scale.is_zero() {
        return Some(ctx.num(0));
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    if denominator_poly.is_zero() {
        return None;
    }

    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let mut numerator_poly = denominator_poly.mul(&radicand_poly.derivative()).add(
        &radicand_poly
            .mul(&denominator_poly.derivative())
            .mul(&two_poly),
    );
    if numerator_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let mut presentation_denominator_power = None;
    if let Some((base, exponent, base_poly)) = denominator_power_factor {
        if exponent > 1 {
            let cancellable = polynomial_power_for_calculus_presentation(&base_poly, exponent - 1);
            if let Ok((quotient, remainder)) = numerator_poly.div_rem(&cancellable) {
                if remainder.is_zero() {
                    numerator_poly = quotient;
                    let denominator_exponent = ctx.num((exponent + 1) as i64);
                    presentation_denominator_power =
                        Some(ctx.add(Expr::Pow(base, denominator_exponent)));
                }
            }
        }
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let raw_denominator = denominator_poly.to_expr(ctx);
    let (denominator_core, denominator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_denominator);
    let denominator_content_square = denominator_content.clone() * denominator_content;
    if denominator_content_square.is_zero() {
        return None;
    }

    let coefficient = -numerator_scale * numerator_content * BigRational::new(1.into(), 2.into())
        / denominator_content_square;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_parts = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_parts.push(radicand);
    denominator_parts.push(sqrt_radicand);
    if !cas_ast::views::as_rational_const(ctx, denominator_core, 8)
        .is_some_and(|value| value.is_one())
    {
        denominator_parts.push(
            presentation_denominator_power.unwrap_or_else(|| squared_expr(ctx, denominator_core)),
        );
    }
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(crate) fn reciprocal_sqrt_polynomial_product_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand =
        if let Some((_, radicand, _)) = negative_half_power_target_parts(ctx, target, var_name) {
            radicand
        } else {
            let Expr::Div(_, denominator_expr) = ctx.get(target).clone() else {
                return None;
            };
            let (radicand, _) = denominator_sqrt_polynomial_parts(ctx, denominator_expr, var_name)?;
            radicand
        };
    let result = reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, var_name)?;
    Some((result, vec![crate::ImplicitCondition::Positive(radicand)]))
}

fn sqrt_denominator_positive_integer_power_factor(
    ctx: &mut Context,
    denominator_expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, usize, Polynomial)> {
    let mut power_factor = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, denominator_expr) {
        if extract_square_root_base(ctx, factor).is_some() {
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_some() {
            continue;
        }
        let factor_power =
            positive_integer_polynomial_power_for_calculus_presentation(ctx, factor, var_name)?;
        if power_factor.replace(factor_power).is_some() {
            return None;
        }
    }
    power_factor
}

pub(super) fn inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => -BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => BigRational::one(),
        _ => return None,
    };

    let Expr::Div(numerator_expr, denominator_expr) = ctx.get(args[0]).clone() else {
        return None;
    };
    let argument_scale = cas_ast::views::as_rational_const(ctx, numerator_expr, 8)?;
    if argument_scale.is_zero() {
        return None;
    }
    let (radicand, denominator_poly) =
        denominator_sqrt_polynomial_parts(ctx, denominator_expr, var_name)?;
    if denominator_poly.is_zero() || denominator_poly == Polynomial::one(var_name.to_string()) {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let numerator_poly = denominator_poly.mul(&radicand_poly.derivative()).add(
        &radicand_poly
            .mul(&denominator_poly.derivative())
            .mul(&two_poly),
    );
    if numerator_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let raw_denominator = denominator_poly.to_expr(ctx);
    let (denominator_core, denominator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_denominator);
    let denominator_content_square = denominator_content.clone() * denominator_content;
    if denominator_content_square.is_zero() {
        return None;
    }

    let coefficient = derivative_sign
        * argument_scale.clone()
        * numerator_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator_square = squared_expr(ctx, denominator_core);
    let root_product =
        cas_math::expr_nary::build_balanced_mul(ctx, &[radicand, denominator_square]);
    let scaled_root_product =
        scale_expr_for_calculus_presentation(ctx, denominator_content_square, root_product);
    let scale_square = argument_scale.clone() * argument_scale;
    let scale_square_expr = rational_const_for_calculus_presentation(ctx, scale_square);
    let gap = ctx.add(Expr::Add(scaled_root_product, scale_square_expr));

    let mut denominator_parts = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_parts.push(sqrt_radicand);
    denominator_parts.push(gap);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);

    let required_conditions = positive_polynomial_radicand_and_nonzero_required_conditions(
        radicand,
        &radicand_poly,
        denominator_core,
    );

    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some((ctx.add(Expr::Hold(compact)), required_conditions))
}

pub(super) fn constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, inner) = rational_scaled_single_factor_allow_unit(ctx, target)?;
    let (derivative, required_conditions) =
        inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
            ctx, inner, var_name,
        )
        .or_else(|| {
            inverse_tangent_reciprocal_sqrt_shifted_sqrt_product_derivative_presentation(
                ctx, inner, var_name,
            )
        })?;
    let derivative = if scale.is_one() {
        unwrap_internal_hold_for_calculus(ctx, derivative)
    } else {
        scale_compact_derivative_by_rational(ctx, derivative, scale)
    };
    Some((ctx.add(Expr::Hold(derivative)), required_conditions))
}

pub(super) fn constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) =
        constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
            ctx,
            target,
            &call.var_name,
        )?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        required_conditions,
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{
        constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation,
        constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_rewrite,
    };
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn constant_scaled_reciprocal_sqrt_product_arctan_derivative_reuses_compact_route() {
        let mut ctx = Context::new();
        let expr = parse("2*arctan(1/(sqrt(x)*(x+1)))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_presentation(
                &mut ctx, expr, "x",
            )
            .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(3 * x + 1) / ((x * (x + 1)^2 + 1) * sqrt(x))"
        );
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn constant_scaled_reciprocal_sqrt_product_arctan_rewrite_preserves_conditions() {
        let mut ctx = Context::new();
        let target = parse("2*arctan(1/(sqrt(x)*(x+1)))", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = constant_scaled_inverse_tangent_reciprocal_sqrt_product_derivative_rewrite(
            &mut ctx, &call, target,
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "-(3 * x + 1) / ((x * (x + 1)^2 + 1) * sqrt(x))"
        );
        assert_eq!(rewrite.required_conditions.len(), 2);
    }
}

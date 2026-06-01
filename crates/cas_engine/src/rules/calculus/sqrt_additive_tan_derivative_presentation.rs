use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::Zero;

use super::differentiation::differentiate;
use super::elementary_variable_term_presentation::{
    scaled_exp_bounded_chain_derivative_for_calculus_presentation,
    scaled_exp_variable_term_for_calculus_presentation,
    scaled_ln_variable_arg_for_calculus_presentation,
    scaled_reciprocal_sqrt_variable_term_for_calculus_presentation,
    scaled_sec_or_csc_variable_derivative_for_calculus_presentation,
    scaled_sqrt_variable_term_for_calculus_presentation,
    scaled_tan_or_cot_variable_arg_for_calculus_presentation,
};
use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_compaction::{
    bounded_sin_cos_term_bound_for_calculus_presentation,
    compact_numeric_mul_factors_for_calculus_presentation,
};
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};
use super::sqrt_additive_result_presentation::{
    reciprocal_sqrt_derivative_term_for_calculus_presentation,
    sqrt_additive_generic_common_denominator_derivative_presentation,
    sqrt_additive_generic_common_denominator_reciprocal_sqrt_variable_derivative_presentation,
    sqrt_additive_generic_common_denominator_sqrt_and_reciprocal_sqrt_variable_derivative_presentation,
    sqrt_additive_generic_common_denominator_sqrt_variable_derivative_presentation,
    sqrt_additive_generic_derivative_presentation,
    sqrt_additive_generic_reciprocal_sqrt_variable_derivative_presentation,
    sqrt_additive_generic_sqrt_and_reciprocal_sqrt_variable_derivative_presentation,
    sqrt_additive_generic_sqrt_variable_derivative_presentation,
    sqrt_variable_derivative_term_for_calculus_presentation,
};
use super::sqrt_additive_tan_result_presentation::{
    compact_tan_sqrt_common_denominator_numerator_term,
    sqrt_additive_tan_reciprocal_sqrt_variable_derivative_presentation,
    sqrt_additive_tan_sqrt_and_reciprocal_sqrt_variable_derivative_presentation,
    sqrt_additive_tan_sqrt_variable_derivative_presentation,
    SqrtAdditiveTanDerivativePresentationParts,
};

pub(crate) fn sqrt_additive_tan_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut tan_scale = BigRational::zero();
    let mut tan_arg = None;
    let mut common_trig_denominator_builtin = None;
    let mut has_variable_dependency = false;
    let mut common_denominator = None;
    let mut sqrt_variable_derivative = None;
    let mut reciprocal_sqrt_variable_derivative = None;
    let mut reciprocal_derivative_scales = Vec::new();
    let mut other_derivatives = Vec::new();
    let mut has_reciprocal_trig_term = false;
    let mut required_conditions = Vec::new();
    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if let Some((scale, arg, denominator_builtin)) =
            scaled_tan_or_cot_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if tan_arg.is_some_and(|existing| existing != arg) {
                return None;
            }
            if common_trig_denominator_builtin
                .is_some_and(|existing| existing != denominator_builtin)
            {
                return None;
            }
            tan_arg = Some(arg);
            common_trig_denominator_builtin = Some(denominator_builtin);
            tan_scale += scale;
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                other_derivatives.push(derivative);
            }
            continue;
        }
        if let Some((derivative, required_condition)) =
            scaled_sec_or_csc_variable_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            has_reciprocal_trig_term = true;
            other_derivatives.push(derivative);
            required_conditions.push(required_condition);
            continue;
        }
        if let Some((ln_scale, ln_arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if common_denominator.is_some_and(|existing| existing != ln_arg) {
                return None;
            }
            common_denominator = Some(ln_arg);
            reciprocal_derivative_scales.push(ln_scale);
            required_conditions.push(crate::ImplicitCondition::Positive(ln_arg));
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if let Some((existing_scale, existing_arg)) = &mut sqrt_variable_derivative {
                if *existing_arg != sqrt_arg {
                    return None;
                }
                *existing_scale += sqrt_scale;
            } else {
                sqrt_variable_derivative = Some((sqrt_scale, sqrt_arg));
            }
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            match reciprocal_sqrt_variable_derivative.take() {
                Some((mut existing_scale, existing_arg)) if existing_arg == reciprocal_sqrt_arg => {
                    existing_scale += reciprocal_sqrt_scale;
                    reciprocal_sqrt_variable_derivative = Some((existing_scale, existing_arg));
                }
                Some((previous_scale, previous_arg)) => {
                    other_derivatives.push(
                        reciprocal_sqrt_derivative_term_for_calculus_presentation(
                            ctx,
                            previous_scale,
                            previous_arg,
                        ),
                    );
                    other_derivatives.push(
                        reciprocal_sqrt_derivative_term_for_calculus_presentation(
                            ctx,
                            reciprocal_sqrt_scale,
                            reciprocal_sqrt_arg,
                        ),
                    );
                }
                None => {
                    reciprocal_sqrt_variable_derivative =
                        Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg));
                }
            }
            required_conditions.push(crate::ImplicitCondition::Positive(reciprocal_sqrt_arg));
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            other_derivatives.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some(exp_chain_derivative) =
            scaled_exp_bounded_chain_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            other_derivatives.push(exp_chain_derivative);
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            other_derivatives.push(derivative.to_expr(ctx));
        }
    }

    if !has_variable_dependency {
        return None;
    }
    if tan_scale.is_zero() {
        let has_common_denominator_sqrt_and_reciprocal_sqrt_route = common_denominator.is_some()
            && sqrt_variable_derivative.is_some()
            && reciprocal_sqrt_variable_derivative.is_some()
            && !reciprocal_derivative_scales.is_empty()
            && !other_derivatives.is_empty();
        if !has_reciprocal_trig_term && !has_common_denominator_sqrt_and_reciprocal_sqrt_route {
            return None;
        }
        if let Some((sqrt_scale, sqrt_arg)) = sqrt_variable_derivative {
            if let Some(common_denominator) = common_denominator {
                if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
                    reciprocal_sqrt_variable_derivative
                {
                    if reciprocal_sqrt_arg == sqrt_arg
                        && sqrt_arg == common_denominator
                        && !sqrt_scale.is_zero()
                        && !reciprocal_sqrt_scale.is_zero()
                        && !reciprocal_derivative_scales.is_empty()
                        && !other_derivatives.is_empty()
                    {
                        let result =
                            sqrt_additive_generic_common_denominator_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                                ctx,
                                radicand,
                                common_denominator,
                                sqrt_scale,
                                reciprocal_sqrt_scale,
                                reciprocal_derivative_scales,
                                other_derivatives,
                            )?;
                        return Some((result, radicand, required_conditions));
                    }
                    return None;
                } else if !reciprocal_derivative_scales.is_empty() && !other_derivatives.is_empty()
                {
                    let result =
                        sqrt_additive_generic_common_denominator_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            common_denominator,
                            sqrt_arg,
                            sqrt_scale,
                            reciprocal_derivative_scales,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                return None;
            }
            if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
                reciprocal_sqrt_variable_derivative
            {
                if reciprocal_sqrt_arg == sqrt_arg
                    && !sqrt_scale.is_zero()
                    && !reciprocal_sqrt_scale.is_zero()
                {
                    let result =
                        sqrt_additive_generic_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            sqrt_arg,
                            sqrt_scale,
                            reciprocal_sqrt_scale,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                    ctx,
                    reciprocal_sqrt_scale,
                    reciprocal_sqrt_arg,
                ));
            }
            if has_reciprocal_trig_term {
                let mut derivative_terms = other_derivatives;
                derivative_terms.push(sqrt_variable_derivative_term_for_calculus_presentation(
                    ctx, sqrt_scale, sqrt_arg,
                )?);
                let result =
                    sqrt_additive_generic_derivative_presentation(ctx, radicand, derivative_terms)?;
                return Some((result, radicand, required_conditions));
            }
            let result = sqrt_additive_generic_sqrt_variable_derivative_presentation(
                ctx,
                radicand,
                sqrt_arg,
                sqrt_scale,
                other_derivatives,
            )?;
            return Some((result, radicand, required_conditions));
        }

        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            reciprocal_sqrt_variable_derivative
        {
            if common_denominator.is_none() && !reciprocal_sqrt_scale.is_zero() {
                let result =
                    sqrt_additive_generic_reciprocal_sqrt_variable_derivative_presentation(
                        ctx,
                        radicand,
                        reciprocal_sqrt_arg,
                        reciprocal_sqrt_scale,
                        other_derivatives,
                    )?;
                return Some((result, radicand, required_conditions));
            }
            if let Some(common_denominator) = common_denominator {
                if reciprocal_sqrt_arg == common_denominator
                    && !reciprocal_sqrt_scale.is_zero()
                    && !reciprocal_derivative_scales.is_empty()
                    && !other_derivatives.is_empty()
                {
                    let result =
                        sqrt_additive_generic_common_denominator_reciprocal_sqrt_variable_derivative_presentation(
                            ctx,
                            radicand,
                            common_denominator,
                            reciprocal_sqrt_scale,
                            reciprocal_derivative_scales,
                            other_derivatives,
                        )?;
                    return Some((result, radicand, required_conditions));
                }
                other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                    ctx,
                    reciprocal_sqrt_scale,
                    reciprocal_sqrt_arg,
                ));
            }
        }

        if common_denominator.is_none()
            && reciprocal_derivative_scales.is_empty()
            && !other_derivatives.is_empty()
        {
            let result =
                sqrt_additive_generic_derivative_presentation(ctx, radicand, other_derivatives)?;
            return Some((result, radicand, required_conditions));
        }
        if let Some(common_denominator) = common_denominator {
            if !reciprocal_derivative_scales.is_empty() && !other_derivatives.is_empty() {
                let result = sqrt_additive_generic_common_denominator_derivative_presentation(
                    ctx,
                    radicand,
                    common_denominator,
                    reciprocal_derivative_scales,
                    other_derivatives,
                )?;
                return Some((result, radicand, required_conditions));
            }
        }

        return None;
    }
    let tan_arg = tan_arg?;
    let common_trig_denominator_builtin = common_trig_denominator_builtin?;
    let reciprocal_trig_builtin = match common_trig_denominator_builtin {
        BuiltinFn::Cos => BuiltinFn::Sec,
        BuiltinFn::Sin => BuiltinFn::Csc,
        _ => return None,
    };
    let cos_arg = ctx.call_builtin(common_trig_denominator_builtin, vec![tan_arg]);
    let two = ctx.num(2);
    let cos_square = ctx.add_raw(Expr::Pow(cos_arg, two));

    if sqrt_variable_derivative.is_none()
        && reciprocal_sqrt_variable_derivative.is_none()
        && common_denominator.is_some()
        && matches!(reciprocal_derivative_scales.as_slice(), [scale] if !scale.is_zero())
    {
        let (result, _, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_inline_presentation(ctx, target, var_name)?;
        return Some((result, radicand, required_conditions));
    }

    if let Some((sqrt_scale, sqrt_arg)) = sqrt_variable_derivative {
        if common_denominator.is_some() {
            return None;
        }
        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            reciprocal_sqrt_variable_derivative
        {
            if reciprocal_sqrt_arg == sqrt_arg
                && !sqrt_scale.is_zero()
                && !reciprocal_sqrt_scale.is_zero()
            {
                let result =
                    sqrt_additive_tan_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
                        ctx,
                        SqrtAdditiveTanDerivativePresentationParts {
                            radicand,
                            tan_arg,
                            reciprocal_trig_builtin,
                            tan_scale: tan_scale.clone(),
                            other_derivatives,
                        },
                        sqrt_arg,
                        sqrt_scale,
                        reciprocal_sqrt_scale,
                    )?;
                required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
                return Some((result, radicand, required_conditions));
            }
            other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                ctx,
                reciprocal_sqrt_scale,
                reciprocal_sqrt_arg,
            ));
        }
        let result = sqrt_additive_tan_sqrt_variable_derivative_presentation(
            ctx,
            SqrtAdditiveTanDerivativePresentationParts {
                radicand,
                tan_arg,
                reciprocal_trig_builtin,
                tan_scale: tan_scale.clone(),
                other_derivatives,
            },
            sqrt_arg,
            sqrt_scale,
        )?;
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        return Some((result, radicand, required_conditions));
    }

    if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) = reciprocal_sqrt_variable_derivative
    {
        if common_denominator.is_none() && !reciprocal_sqrt_scale.is_zero() {
            let result = sqrt_additive_tan_reciprocal_sqrt_variable_derivative_presentation(
                ctx,
                SqrtAdditiveTanDerivativePresentationParts {
                    radicand,
                    tan_arg,
                    reciprocal_trig_builtin,
                    tan_scale: tan_scale.clone(),
                    other_derivatives,
                },
                reciprocal_sqrt_arg,
                reciprocal_sqrt_scale,
            )?;
            required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
            return Some((result, radicand, required_conditions));
        }
        other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
            ctx,
            reciprocal_sqrt_scale,
            reciprocal_sqrt_arg,
        ));
    }

    if common_denominator.is_none() && reciprocal_derivative_scales.is_empty() {
        let reciprocal_trig_arg = ctx.call_builtin(reciprocal_trig_builtin, vec![tan_arg]);
        let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
        let tan_derivative =
            scale_expr_for_calculus_presentation(ctx, tan_scale.clone(), reciprocal_trig_square);
        other_derivatives.insert(0, tan_derivative);
        let result =
            sqrt_additive_generic_derivative_presentation(ctx, radicand, other_derivatives)?;
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        return Some((result, radicand, required_conditions));
    }

    let mut numerator_terms = Vec::new();
    let common_denominator =
        common_denominator.filter(|_| !reciprocal_derivative_scales.is_empty());
    let tan_numerator = if let Some(denominator) = common_denominator {
        scale_expr_for_calculus_presentation(ctx, tan_scale, denominator)
    } else {
        rational_const_for_calculus_presentation(ctx, tan_scale)
    };
    numerator_terms.push(tan_numerator);
    for scale in reciprocal_derivative_scales {
        numerator_terms.push(scale_expr_for_calculus_presentation(ctx, scale, cos_square));
    }
    for derivative in other_derivatives {
        let mut term = compact_tan_sqrt_common_denominator_numerator_term(
            ctx, cos_arg, cos_square, derivative,
        );
        if let Some(denominator) = common_denominator {
            term = ctx.add_raw(Expr::Mul(denominator, term));
            term = compact_numeric_mul_factors_for_calculus_presentation(ctx, term);
        }
        numerator_terms.push(term);
    }
    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let denominator_scale =
        rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if let Some(common_denominator) = common_denominator {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[
                denominator_scale,
                common_denominator,
                cos_square,
                sqrt_radicand,
            ],
        )
    } else {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[denominator_scale, cos_square, sqrt_radicand],
        )
    };
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some((cas_ast::hold::wrap_hold(ctx, compact), radicand, {
        required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
        required_conditions
    }))
}

pub(crate) fn sqrt_additive_tan_polynomial_derivative_inline_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let terms = cas_math::expr_nary::add_terms_signed(ctx, radicand);
    if terms.len() < 2 || terms.len() > 6 {
        return None;
    }

    let mut tan_scale = BigRational::zero();
    let mut tan_arg = None;
    let mut common_trig_denominator_builtin = None;
    let mut sqrt_variable_derivative = None;
    let mut other_derivatives = Vec::new();
    let mut has_variable_dependency = false;
    let mut has_ln_derivative = false;
    let mut has_reciprocal_sqrt_derivative = false;
    let mut required_conditions = Vec::new();

    for (term, sign) in terms {
        let signed_term = if sign == cas_math::expr_nary::Sign::Neg {
            ctx.add(Expr::Neg(term))
        } else {
            term
        };
        has_variable_dependency |= contains_named_var(ctx, signed_term, var_name);

        if let Some((scale, arg, denominator_builtin)) =
            scaled_tan_or_cot_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if tan_arg.is_some_and(|existing| existing != arg) {
                return None;
            }
            if common_trig_denominator_builtin
                .is_some_and(|existing| existing != denominator_builtin)
            {
                return None;
            }
            tan_arg = Some(arg);
            common_trig_denominator_builtin = Some(denominator_builtin);
            tan_scale += scale;
            continue;
        }
        if cas_ast::views::as_rational_const(ctx, signed_term, 8).is_some() {
            continue;
        }
        if let Some((sqrt_scale, sqrt_arg)) =
            scaled_sqrt_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            if let Some((existing_scale, existing_arg)) = &mut sqrt_variable_derivative {
                if *existing_arg != sqrt_arg {
                    return None;
                }
                *existing_scale += sqrt_scale;
            } else {
                sqrt_variable_derivative = Some((sqrt_scale, sqrt_arg));
            }
            required_conditions.push(crate::ImplicitCondition::Positive(sqrt_arg));
            continue;
        }
        if let Some((ln_scale, ln_arg)) =
            scaled_ln_variable_arg_for_calculus_presentation(ctx, signed_term, var_name)
        {
            has_ln_derivative |= !ln_scale.is_zero();
            let numerator = rational_const_for_calculus_presentation(ctx, ln_scale);
            let reciprocal = ctx.add_raw(Expr::Div(numerator, ln_arg));
            other_derivatives.push(reciprocal);
            required_conditions.push(crate::ImplicitCondition::Positive(ln_arg));
            continue;
        }
        if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) =
            scaled_reciprocal_sqrt_variable_term_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            if has_reciprocal_sqrt_derivative {
                return None;
            }
            has_reciprocal_sqrt_derivative = true;
            if !reciprocal_sqrt_scale.is_zero() {
                other_derivatives.push(reciprocal_sqrt_derivative_term_for_calculus_presentation(
                    ctx,
                    reciprocal_sqrt_scale,
                    reciprocal_sqrt_arg,
                ));
            }
            required_conditions.push(crate::ImplicitCondition::Positive(reciprocal_sqrt_arg));
            continue;
        }
        if bounded_sin_cos_term_bound_for_calculus_presentation(ctx, signed_term).is_some() {
            let derivative = differentiate(ctx, signed_term, var_name)?;
            if !cas_ast::views::as_rational_const(ctx, derivative, 8)
                .is_some_and(|value| value.is_zero())
            {
                other_derivatives.push(derivative);
            }
            continue;
        }
        if let Some((exp_scale, exp_term)) =
            scaled_exp_variable_term_for_calculus_presentation(ctx, signed_term, var_name)
        {
            other_derivatives.push(scale_expr_for_calculus_presentation(
                ctx, exp_scale, exp_term,
            ));
            continue;
        }
        if let Some(exp_chain_derivative) =
            scaled_exp_bounded_chain_derivative_for_calculus_presentation(
                ctx,
                signed_term,
                var_name,
            )
        {
            other_derivatives.push(exp_chain_derivative);
            continue;
        }
        let poly = polynomial_radicand_for_calculus_presentation(ctx, signed_term, var_name)?;
        if poly.degree() > 3 || poly.coeffs.len() > 5 {
            return None;
        }
        let derivative = poly.derivative();
        if !derivative.is_zero() {
            other_derivatives.push(derivative.to_expr(ctx));
        }
    }

    if !has_variable_dependency || tan_scale.is_zero() {
        return None;
    }
    if sqrt_variable_derivative.is_none() && !has_ln_derivative {
        return None;
    }
    if sqrt_variable_derivative.is_some() && has_reciprocal_sqrt_derivative {
        return None;
    }
    if sqrt_variable_derivative
        .as_ref()
        .is_some_and(|(sqrt_scale, _)| sqrt_scale.is_zero())
    {
        return None;
    }
    let tan_arg = tan_arg?;
    let common_trig_denominator_builtin = common_trig_denominator_builtin?;
    let reciprocal_trig_builtin = match common_trig_denominator_builtin {
        BuiltinFn::Cos => BuiltinFn::Sec,
        BuiltinFn::Sin => BuiltinFn::Csc,
        _ => return None,
    };

    let cos_arg = ctx.call_builtin(common_trig_denominator_builtin, vec![tan_arg]);
    let two = ctx.num(2);
    let reciprocal_trig_arg = ctx.call_builtin(reciprocal_trig_builtin, vec![tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let mut derivative_terms = Vec::new();
    derivative_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        tan_scale,
        reciprocal_trig_square,
    ));
    if let Some((sqrt_scale, sqrt_arg)) = sqrt_variable_derivative {
        derivative_terms.push(sqrt_variable_derivative_term_for_calculus_presentation(
            ctx, sqrt_scale, sqrt_arg,
        )?);
    }
    derivative_terms.extend(other_derivatives);

    let result = sqrt_additive_generic_derivative_presentation(ctx, radicand, derivative_terms)?;
    required_conditions.push(crate::ImplicitCondition::NonZero(cos_arg));
    Some((result, radicand, required_conditions))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::arctan_sqrt_additive_derivative_presentation::arctan_sqrt_additive_tan_polynomial_derivative_presentation;
    use super::*;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn sqrt_additive_tan_exp_polynomial_derivative_presentation_accepts_exp_term() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+exp(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(e^x + sec(x)^2 + 1) / (2 * sqrt(tan(x) + e^x + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + e^x + x");
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn sqrt_additive_tan_cos_square_polynomial_derivative_compacts_power_exponent() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+cos(x)^2+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(sec(x)^2 + 1 - 2 * cos(x) * sin(x)) / (2 * sqrt(tan(x) + cos(x)^2 + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + cos(x)^2 + x");
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn sqrt_additive_tan_ln_polynomial_derivative_inline_presentation_accepts_log_term() {
        for (
            input,
            expected_result,
            expected_radicand,
            expected_required_conditions_len,
        ) in [
            (
                "sqrt(tan(x)+ln(x)+x)",
                "(sec(x)^2 + 1 / x + 1) / (2 * sqrt(tan(x) + ln(x) + x))",
                "tan(x) + ln(x) + x",
                2,
            ),
            (
                "sqrt(tan(x)+2*ln(x)+x)",
                "(sec(x)^2 + 2 / x + 1) / (2 * sqrt(tan(x) + 2 * ln(x) + x))",
                "tan(x) + 2 * ln(x) + x",
                2,
            ),
            (
                "sqrt(tan(x)-ln(x)+x)",
                "(sec(x)^2 + 1 - 1 / x) / (2 * sqrt(tan(x) - ln(x) + x))",
                "tan(x) - ln(x) + x",
                2,
            ),
            (
                "sqrt(tan(x)+ln(x)+sqrt(x)+x)",
                "(sec(x)^2 + 1 / x + 1 / (2 * sqrt(x)) + 1) / (2 * sqrt(tan(x) + ln(x) + sqrt(x) + x))",
                "tan(x) + ln(x) + sqrt(x) + x",
                3,
            ),
            (
                "sqrt(tan(x)+ln(x)+1/sqrt(x)+x)",
                "(sec(x)^2 + 1 / x + 1 - 1/2 * x^(-3/2)) / (2 * sqrt(tan(x) + ln(x) + 1 / sqrt(x) + x))",
                "tan(x) + ln(x) + 1 / sqrt(x) + x",
                3,
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_inline_presentation(&mut ctx, target, "x")
                    .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(
                rendered(&ctx, radicand),
                expected_radicand,
                "input: {input}"
            );
            assert_eq!(
                required_conditions.len(),
                expected_required_conditions_len,
                "input: {input}"
            );
        }
    }

    #[test]
    fn sqrt_additive_tan_exp_linear_polynomial_derivative_presentation_accepts_chain_factor() {
        for (input, expected_result, expected_radicand) in [
            (
                "sqrt(tan(x)+exp(2*x)+x)",
                "(sec(x)^2 + 2 * e^(2 * x) + 1) / (2 * sqrt(tan(x) + e^(2 * x) + x))",
                "tan(x) + e^(2 * x) + x",
            ),
            (
                "sqrt(tan(x)+exp(2*x+1)+x)",
                "(sec(x)^2 + 2 * e^(2 * x + 1) + 1) / (2 * sqrt(tan(x) + e^(2 * x + 1) + x))",
                "tan(x) + e^(2 * x + 1) + x",
            ),
            (
                "sqrt(tan(x)+exp(-2*x)+x)",
                "(sec(x)^2 + 1 - 2 * e^(-2 * x)) / (2 * sqrt(tan(x) + e^(-2 * x) + x))",
                "tan(x) + e^(-2 * x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x")
                    .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(
                rendered(&ctx, radicand),
                expected_radicand,
                "input: {input}"
            );
            assert_eq!(required_conditions.len(), 1, "input: {input}");
        }
    }

    #[test]
    fn sqrt_additive_tan_reciprocal_sqrt_derivative_presentation_accepts_inverse_sqrt_term() {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)+1/sqrt(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + 1 / sqrt(x) + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) + 1 / sqrt(x) + x");
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn sqrt_additive_tan_negative_reciprocal_sqrt_derivative_presentation_accepts_signed_inverse_sqrt_term(
    ) {
        let mut ctx = Context::new();
        let target = parse("sqrt(tan(x)-1/sqrt(x)+x)", &mut ctx).unwrap();
        let (result, radicand, required_conditions) =
            sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 1) / (4 * x * sqrt(x) * sqrt(tan(x) - 1 / sqrt(x) + x))"
        );
        assert_eq!(rendered(&ctx, radicand), "tan(x) - 1 / sqrt(x) + x");
        assert_eq!(required_conditions.len(), 2);
    }

    #[test]
    fn sqrt_additive_tan_mixed_sqrt_and_reciprocal_sqrt_derivative_presentation_uses_common_denominator(
    ) {
        for (input, expected_result, expected_radicand) in [
            (
                "sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x)",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + x - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + sqrt(x) + 1 / sqrt(x) + x))",
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x",
            ),
            (
                "sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x)",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 2 * x + 3) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x))",
                "tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                sqrt_additive_tan_polynomial_derivative_presentation(&mut ctx, target, "x")
                    .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(
                rendered(&ctx, radicand),
                expected_radicand,
                "input: {input}"
            );
            assert_eq!(required_conditions.len(), 3, "input: {input}");
        }
    }

    #[test]
    fn arctan_sqrt_additive_tan_mixed_sqrt_derivative_presentation_reuses_inner_common_denominator()
    {
        for (input, expected_result, expected_radicand) in [
            (
                "arctan(sqrt(tan(x)+sqrt(x)+1/sqrt(x)+x))",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + x - 1) / (4 * x * sqrt(x) * sqrt(tan(x) + sqrt(x) + 1 / sqrt(x) + x) * (tan(x) + sqrt(x) + 1 / sqrt(x) + x + 1))",
                "tan(x) + sqrt(x) + 1 / sqrt(x) + x",
            ),
            (
                "arctan(sqrt(tan(x)+2*sqrt(x)-3/sqrt(x)+x))",
                "(2 * x * sqrt(x) + 2 * x * sqrt(x) * sec(x)^2 + 2 * x + 3) / (4 * x * sqrt(x) * sqrt(tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x) * (tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x + 1))",
                "tan(x) + 2 * sqrt(x) - 3 / sqrt(x) + x",
            ),
        ] {
            let mut ctx = Context::new();
            let target = parse(input, &mut ctx).unwrap();
            let (result, radicand, required_conditions) =
                arctan_sqrt_additive_tan_polynomial_derivative_presentation(
                    &mut ctx, target, "x",
                )
                .unwrap();

            assert_eq!(rendered(&ctx, result), expected_result, "input: {input}");
            assert_eq!(
                rendered(&ctx, radicand),
                expected_radicand,
                "input: {input}"
            );
            assert_eq!(required_conditions.len(), 3, "input: {input}");
        }
    }
}

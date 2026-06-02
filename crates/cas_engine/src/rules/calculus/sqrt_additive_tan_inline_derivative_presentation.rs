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
    scaled_sqrt_variable_term_for_calculus_presentation,
    scaled_tan_or_cot_variable_arg_for_calculus_presentation,
};
use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_compaction::bounded_sin_cos_term_bound_for_calculus_presentation;
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};
use super::sqrt_additive_result_presentation::{
    reciprocal_sqrt_derivative_term_for_calculus_presentation,
    sqrt_additive_generic_derivative_presentation,
    sqrt_variable_derivative_term_for_calculus_presentation,
};

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

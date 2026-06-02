use cas_ast::{Context, ExprId};
use num_rational::BigRational;
use num_traits::Zero;

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

pub(super) struct SqrtAdditiveGenericDerivativeRoute {
    pub(super) radicand: ExprId,
    pub(super) common_denominator: Option<ExprId>,
    pub(super) sqrt_variable_derivative: Option<(BigRational, ExprId)>,
    pub(super) reciprocal_sqrt_variable_derivative: Option<(BigRational, ExprId)>,
    pub(super) reciprocal_derivative_scales: Vec<BigRational>,
    pub(super) other_derivatives: Vec<ExprId>,
    pub(super) has_reciprocal_trig_term: bool,
    pub(super) required_conditions: Vec<crate::ImplicitCondition>,
}

pub(super) fn try_sqrt_additive_generic_derivative_route(
    ctx: &mut Context,
    route: SqrtAdditiveGenericDerivativeRoute,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let SqrtAdditiveGenericDerivativeRoute {
        radicand,
        common_denominator,
        sqrt_variable_derivative,
        reciprocal_sqrt_variable_derivative,
        reciprocal_derivative_scales,
        mut other_derivatives,
        has_reciprocal_trig_term,
        required_conditions,
    } = route;

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
            } else if !reciprocal_derivative_scales.is_empty() && !other_derivatives.is_empty() {
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

    if let Some((reciprocal_sqrt_scale, reciprocal_sqrt_arg)) = reciprocal_sqrt_variable_derivative
    {
        if common_denominator.is_none() && !reciprocal_sqrt_scale.is_zero() {
            let result = sqrt_additive_generic_reciprocal_sqrt_variable_derivative_presentation(
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

    None
}

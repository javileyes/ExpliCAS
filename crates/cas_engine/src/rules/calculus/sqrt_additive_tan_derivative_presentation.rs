use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_traits::Zero;

use super::scalar_presentation::scale_expr_for_calculus_presentation;
use super::sqrt_additive_generic_derivative_routes::{
    try_sqrt_additive_generic_derivative_route, SqrtAdditiveGenericDerivativeRoute,
};
use super::sqrt_additive_result_presentation::{
    reciprocal_sqrt_derivative_term_for_calculus_presentation,
    sqrt_additive_generic_derivative_presentation,
};
pub(crate) use super::sqrt_additive_tan_inline_derivative_presentation::sqrt_additive_tan_polynomial_derivative_inline_presentation;
use super::sqrt_additive_tan_result_presentation::{
    sqrt_additive_tan_common_denominator_derivative_presentation,
    sqrt_additive_tan_reciprocal_sqrt_variable_derivative_presentation,
    sqrt_additive_tan_sqrt_and_reciprocal_sqrt_variable_derivative_presentation,
    sqrt_additive_tan_sqrt_variable_derivative_presentation,
    SqrtAdditiveTanCommonDenominatorPresentationParts, SqrtAdditiveTanDerivativePresentationParts,
};
use super::sqrt_additive_tan_term_scan::{scan_sqrt_additive_tan_terms, SqrtAdditiveTanTermScan};

pub(crate) fn sqrt_additive_tan_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = extract_square_root_base(ctx, target)?;
    let SqrtAdditiveTanTermScan {
        tan_scale,
        tan_arg,
        common_trig_denominator_builtin,
        common_denominator,
        sqrt_variable_derivative,
        reciprocal_sqrt_variable_derivative,
        reciprocal_derivative_scales,
        mut other_derivatives,
        has_reciprocal_trig_term,
        mut required_conditions,
    } = scan_sqrt_additive_tan_terms(ctx, radicand, var_name)?;

    if tan_scale.is_zero() {
        return try_sqrt_additive_generic_derivative_route(
            ctx,
            SqrtAdditiveGenericDerivativeRoute {
                radicand,
                common_denominator,
                sqrt_variable_derivative,
                reciprocal_sqrt_variable_derivative,
                reciprocal_derivative_scales,
                other_derivatives,
                has_reciprocal_trig_term,
                required_conditions,
            },
        );
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

    let (result, denominator_trig_arg) =
        sqrt_additive_tan_common_denominator_derivative_presentation(
            ctx,
            SqrtAdditiveTanCommonDenominatorPresentationParts {
                radicand,
                tan_arg,
                common_trig_denominator_builtin,
                tan_scale,
                common_denominator,
                reciprocal_derivative_scales,
                other_derivatives,
            },
        );
    Some((result, radicand, {
        required_conditions.push(crate::ImplicitCondition::NonZero(denominator_trig_arg));
        required_conditions
    }))
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

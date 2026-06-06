//! Compact derivative presentation routes kept in `DiffRule` priority order.
//!
//! The individual presentation modules own their detection and formatting
//! policies. This boundary only owns local route ordering and rewrite
//! finalization.

use cas_ast::{Context, ExprId};

use crate::symbolic_calculus_call_support::NamedVarCall;
use crate::{ImplicitCondition, Rewrite};

use super::arctan_by_parts_result_presentation::arctan_affine_by_parts_compact_derivative;
use super::atanh_surd_derivative_presentation::atanh_surd_quotient_compact_derivative;
use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::polynomial_times_sqrt_polynomial_derivative_presentation::polynomial_times_sqrt_polynomial_derivative_presentation;

pub(super) fn compact_derivative_presentation_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) =
        compact_derivative_presentation_with_domain(ctx, target, &call.var_name)?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        required_conditions,
    ))
}

fn compact_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<ImplicitCondition>)> {
    if let Some((result, required_nonzero)) =
        cas_math::symbolic_differentiation_support::rational_positive_quadratic_log_abs_pole_derivative(
            ctx,
            target,
            var_name,
        )
    {
        let required_conditions = required_nonzero
            .into_iter()
            .map(crate::ImplicitCondition::NonZero)
            .collect();
        return Some((result, required_conditions));
    }
    if let Some((result, required_nonzero)) =
        cas_math::symbolic_differentiation_support::rational_separated_linear_log_abs_pole_derivative(
            ctx,
            target,
            var_name,
        )
    {
        let required_conditions = required_nonzero
            .into_iter()
            .map(crate::ImplicitCondition::NonZero)
            .collect();
        return Some((result, required_conditions));
    }
    if let Some((result, required_nonzero)) =
        cas_math::symbolic_differentiation_support::rational_log_ratio_single_pole_derivative(
            ctx, target, var_name,
        )
    {
        let required_conditions = required_nonzero
            .into_iter()
            .map(crate::ImplicitCondition::NonZero)
            .collect();
        return Some((result, required_conditions));
    }

    let result = arctan_affine_by_parts_compact_derivative(ctx, target, var_name)
        .or_else(|| atanh_surd_quotient_compact_derivative(ctx, target, var_name))
        .or_else(|| {
            polynomial_times_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
        })?;
    Some((result, Vec::new()))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use crate::symbolic_calculus_call_support::NamedVarCall;

    use super::super::arctan_by_parts_result_presentation::arctan_affine_by_parts_compact_derivative;
    use super::super::atanh_surd_derivative_presentation::atanh_surd_quotient_compact_derivative;
    use super::super::diff_rule_support::finalize_diff_rewrite_with_conditions;
    use super::super::polynomial_times_sqrt_polynomial_derivative_presentation::polynomial_times_sqrt_polynomial_derivative_presentation;
    use super::compact_derivative_presentation_rewrite;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_presentation_rewrite_preserves_arctan_by_parts_first_route() {
        let mut ctx = Context::new();
        let target = parse(
            "((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3",
            &mut ctx,
        )
        .unwrap();
        let route_result =
            arctan_affine_by_parts_compact_derivative(&mut ctx, target, "x").unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let expected_rewrite = finalize_diff_rewrite_with_conditions(
            &mut ctx,
            &call,
            target,
            route_result,
            Vec::new(),
        );
        let rewrite = compact_derivative_presentation_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, expected_rewrite.new_expr)
        );
        assert_eq!(
            rewrite.required_conditions,
            expected_rewrite.required_conditions
        );
    }

    #[test]
    fn compact_presentation_rewrite_handles_positive_quadratic_log_abs_pole_first() {
        let mut ctx = Context::new();
        let cases = [
            (
                "1/4*ln(x^2+1)-1/2*ln(abs(x-1))-1/(2*(x-1))",
                "1 / (x^4 + 2 * x^2 + 1 - 2 * x^3 - 2 * x)",
            ),
            (
                "1/6*ln(2*x^2+2)-1/2*ln(abs(2*x-2))-1/(2*(2*x-2))",
                "(x^2 + 9 - 2 * x^3 - 2 * x) / (12 * x^4 + 24 * x^2 + 12 - 24 * x^3 - 24 * x)",
            ),
        ];

        for (input, expected) in cases {
            let target = parse(input, &mut ctx).unwrap();
            let call = NamedVarCall {
                target,
                var_name: "x".to_string(),
            };
            let rewrite = compact_derivative_presentation_rewrite(&mut ctx, &call, target).unwrap();

            assert_eq!(rendered(&ctx, rewrite.new_expr), expected);
            assert!(
                rewrite.required_conditions.iter().any(|condition| {
                    matches!(condition, crate::ImplicitCondition::NonZero(expr) if rendered(&ctx, *expr) == "x - 1")
                }),
                "rewrite should preserve the linear pole guard for {input}: {:?}",
                rewrite.required_conditions
            );
        }
    }

    #[test]
    fn compact_presentation_rewrite_handles_log_ratio_single_pole() {
        let mut ctx = Context::new();
        let target = parse("ln(abs(x/(x+1))) + 1/(x+1)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = compact_derivative_presentation_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(rendered(&ctx, rewrite.new_expr), "1 / (x^3 + 2 * x^2 + x)");
        assert!(
            rewrite.required_conditions.iter().any(|condition| {
                matches!(condition, crate::ImplicitCondition::NonZero(expr) if rendered(&ctx, *expr) == "x")
            }),
            "rewrite should preserve the numerator pole guard: {:?}",
            rewrite.required_conditions
        );
        assert!(
            rewrite.required_conditions.iter().any(|condition| {
                matches!(condition, crate::ImplicitCondition::NonZero(expr) if rendered(&ctx, *expr) == "x + 1")
            }),
            "rewrite should preserve the denominator pole guard: {:?}",
            rewrite.required_conditions
        );
    }

    #[test]
    fn compact_presentation_rewrite_handles_scaled_shifted_log_ratio_single_pole() {
        let mut ctx = Context::new();
        let cases = [
            "ln(abs((2*x+3)/(x-5))) + 1/(x-5)",
            "ln(3*abs((2*x+3)/(x-5))) + 1/(x-5)",
        ];

        for input in cases {
            let target = parse(input, &mut ctx).unwrap();
            let call = NamedVarCall {
                target,
                var_name: "x".to_string(),
            };
            let rewrite = compact_derivative_presentation_rewrite(&mut ctx, &call, target).unwrap();

            assert_eq!(
                rendered(&ctx, rewrite.new_expr),
                "(62 - 15 * x) / (2 * x^3 + 20 * x + 75 - 17 * x^2)"
            );
            assert!(
                rewrite.required_conditions.iter().any(|condition| {
                    matches!(condition, crate::ImplicitCondition::NonZero(expr) if rendered(&ctx, *expr) == "2 * x + 3")
                }),
                "rewrite should preserve the scaled numerator pole guard for {input}: {:?}",
                rewrite.required_conditions
            );
            assert!(
                rewrite.required_conditions.iter().any(|condition| {
                    matches!(condition, crate::ImplicitCondition::NonZero(expr) if rendered(&ctx, *expr) == "x - 5")
                }),
                "rewrite should preserve the shifted denominator pole guard for {input}: {:?}",
                rewrite.required_conditions
            );
        }
    }

    #[test]
    fn compact_presentation_rewrite_preserves_atanh_surd_fallback() {
        let mut ctx = Context::new();
        let target = parse("atanh((x^2+x+1)/sqrt(7))", &mut ctx).unwrap();
        let route_result = atanh_surd_quotient_compact_derivative(&mut ctx, target, "x").unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let expected_rewrite = finalize_diff_rewrite_with_conditions(
            &mut ctx,
            &call,
            target,
            route_result,
            Vec::new(),
        );
        let rewrite = compact_derivative_presentation_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, expected_rewrite.new_expr)
        );
        assert_eq!(
            rewrite.required_conditions,
            expected_rewrite.required_conditions
        );
    }

    #[test]
    fn compact_presentation_rewrite_preserves_polynomial_times_sqrt_fallback() {
        let mut ctx = Context::new();
        let target = parse("x*sqrt(x)", &mut ctx).unwrap();
        let route_result =
            polynomial_times_sqrt_polynomial_derivative_presentation(&mut ctx, target, "x")
                .unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let expected_rewrite = finalize_diff_rewrite_with_conditions(
            &mut ctx,
            &call,
            target,
            route_result,
            Vec::new(),
        );
        let rewrite = compact_derivative_presentation_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, expected_rewrite.new_expr)
        );
        assert_eq!(
            rewrite.required_conditions,
            expected_rewrite.required_conditions
        );
    }
}

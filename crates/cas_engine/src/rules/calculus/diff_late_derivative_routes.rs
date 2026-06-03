//! Late fallback diff routes kept in the exact `DiffRule` priority order.
//!
//! This module is intentionally a route group, not a generic matcher registry.
//! Several owned route modules still differ in domain and presentation policy,
//! so this boundary only makes the final fallback order explicit.

use cas_ast::{Context, ExprId};
use num_rational::BigRational;
use num_traits::One;

use crate::symbolic_calculus_call_support::NamedVarCall;
use crate::Rewrite;

use super::acosh_derivative_routes::acosh_direct_derivative_route;
use super::arctan_sqrt_constant_over_polynomial_presentation::arctan_sqrt_constant_over_polynomial_presentation;
use super::arctan_sqrt_quotient_derivative_presentation::arctan_sqrt_positive_polynomial_quotient_derivative_shortcut;
use super::diff_rule_support::{
    finalize_diff_rewrite_with_conditions, sqrt_additive_derivative_shortcut,
};
use super::differentiation::differentiate;
use super::inverse_hyperbolic_scaled_sqrt_derivative_routes::constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route;
use super::inverse_tangent_reciprocal_sqrt_derivative_routes::inverse_tangent_reciprocal_sqrt_derivative_route;
use super::inverse_tangent_trig_affine_derivative_presentation::inverse_tangent_direct_trig_affine_derivative_presentation;
use super::ln_sqrt_derivative_routes::ln_sqrt_derivative_route;
use super::log_sqrt_quotient_derivative_routes::log_sqrt_quotient_derivative_route;
use super::positive_quadratic_derivative_routes::positive_quadratic_derivative_route;
use super::primitive_derivative_routes::primitive_derivative_route;
use super::reciprocal_trig_derivative_presentation::scaled_reciprocal_trig_power_derivative_presentation;
use super::sqrt_bounded_trig_positive_shift_derivative_presentation::sqrt_bounded_trig_positive_shift_derivative_presentation;
use super::sqrt_elementary_derivative_routes::sqrt_elementary_function_derivative_route;
use super::sqrt_trig_log_antiderivative_presentation::sqrt_trig_log_antiderivative_derivative_presentation;

pub(super) fn late_fallback_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let result = constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route(
        ctx,
        target,
        &call.var_name,
    )
    .map(|result| (result, Vec::new()))
    .or_else(|| inverse_tangent_reciprocal_sqrt_derivative_route(ctx, target, &call.var_name))
    .or_else(|| sqrt_trig_log_antiderivative_derivative_presentation(ctx, target, &call.var_name))
    .or_else(|| {
        sqrt_bounded_trig_positive_shift_derivative_presentation(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        let mut required_conditions = Vec::new();
        sqrt_additive_derivative_shortcut(ctx, target, &call.var_name, &mut required_conditions)
            .map(|result| (result, required_conditions))
    })
    .or_else(|| sqrt_elementary_function_derivative_route(ctx, target, &call.var_name))
    .or_else(|| {
        scaled_reciprocal_trig_power_derivative_presentation(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        inverse_tangent_direct_trig_affine_derivative_presentation(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        arctan_sqrt_constant_over_polynomial_presentation(
            ctx,
            target,
            &call.var_name,
            BigRational::one(),
        )
        .map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        arctan_sqrt_positive_polynomial_quotient_derivative_shortcut(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| acosh_direct_derivative_route(ctx, target, &call.var_name))
    .or_else(|| {
        positive_quadratic_derivative_route(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| ln_sqrt_derivative_route(ctx, target, &call.var_name))
    .or_else(|| {
        primitive_derivative_route(ctx, target, &call.var_name).map(|result| (result, Vec::new()))
    })
    .or_else(|| {
        log_sqrt_quotient_derivative_route(ctx, target, &call.var_name)
            .map(|result| (result, Vec::new()))
    })
    .or_else(|| differentiate(ctx, target, &call.var_name).map(|result| (result, Vec::new())))?;
    let (result, shortcut_required_conditions) = result;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        shortcut_required_conditions,
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::inverse_hyperbolic_scaled_sqrt_derivative_routes::constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route;
    use super::late_fallback_derivative_rewrite;
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn late_fallback_rewrite_preserves_first_route_result_and_conditions() {
        let mut ctx = Context::new();
        let target = parse("asinh(sqrt(x))/2", &mut ctx).unwrap();
        let route_result = constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route(
            &mut ctx, target, "x",
        )
        .unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = late_fallback_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, route_result)
        );
        assert!(rewrite.required_conditions.is_empty());
    }
}

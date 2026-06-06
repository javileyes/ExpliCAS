//! Source-side presentation shortcuts for derivatives of supported integrals.
//!
//! This module owns the bounded `diff(integrate(...), x)` shortcut gate. It
//! preserves the route order from `calculus/mod.rs`: each accepted integrand is
//! either verified by the symbolic integrator before being returned, or is an
//! explicitly recognized source-side presentation case whose required
//! conditions are still collected by the caller.
//!
//! Route policy map, in evaluation order:
//! 1. arctan sqrt reciprocal and unit-shift square: verified source return.
//! 2. sqrt-chain logarithm: verified source return.
//! 3. sqrt-chain reciprocal square: verified source return.
//! 4. sqrt-chain reciprocal derivative: verified source return.
//! 5. quadratic times affine logarithm by parts: verified source return.
//! 6. polynomial times arctan affine: source-side presentation return.
//! 7. reciprocal positive quadratic arctan: source-side presentation return.
//! 8. rational linear partial fraction: verified source return.
//! 9. rational linear positive quadratic: verified source return.
//! 10. fractional denominator power substitution: verified source return.
//! 11. sqrt derivative substitution: verified source return.
//! 12. arcsin inverse-sqrt product: source-side held compact/source return.
//! 13. affine/acosh inverse-sqrt product: verified held compact/source return.
//! 14. arcsin/asinh polynomial substitution: verified held source return.
//! 15. positive quadratic powers: verified source return.
//! 16. inverse hyperbolic sqrt reciprocal: verified source return.
//! 17. direct sqrt-trig logarithm compaction: verified compact return.
//! 18. sqrt-trig logarithm presentation: verified compact-only return.
//! 19. sqrt reciprocal trig product: source-side presentation return.
//! 20. direct trig affine argument: verified source return.
//!
//! Keep this map synchronized with the route order. A later extraction should
//! move a whole policy group at a time and preserve whether the returned
//! integrand was independently verified, held for presentation, or accepted as
//! a source-side presentation shortcut.
//!
//! Policy groups:
//! - Routes 1-5, 8-11, and 15-16 are verified source-return routes.
//! - Routes 6, 7, and 19 are source-side presentation exceptions.
//! - Routes 12-14 are held-presentation routes that preserve source or compact
//!   integrands for caller condition collection.
//! - Routes 17-18 are compact-only sqrt-trig-log presentation routes.
//! - Route 20 is the final verified direct-trig fallback.
//!
//! The groups are descriptive, not a registry. Source-order priority remains in
//! the function body.

use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::integral_derivative_arctan_polynomial_routes::arctan_polynomial_source_integral_derivative_shortcut;
use super::integral_derivative_conditioned_return::conditioned_integral_derivative_shortcut_result;
use super::integral_derivative_final_presentation_routes::final_presentation_integral_derivative_route;
use super::integral_derivative_held_presentation_routes::held_presentation_integral_derivative_route;
use super::integral_derivative_presentation_route::SupportedIntegralDerivativePresentationRoute;
use super::integral_derivative_verified_power_inverse_routes::power_inverse_verified_integral_derivative_route;
use super::integral_derivative_verified_rational_substitution_routes::rational_substitution_verified_integral_derivative_route;
use super::integral_derivative_verified_source_routes::initial_verified_source_integral_derivative_route;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{try_extract_integrate_call, NamedVarCall};
use cas_ast::{Context, ExprId};

pub(super) fn supported_integral_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let integrate_call = try_extract_integrate_call(ctx, target)?;
    supported_integral_derivative_presentation_for_integrate_call(ctx, &integrate_call, var_name)
}

fn supported_integral_derivative_presentation_for_integrate_call(
    ctx: &mut Context,
    integrate_call: &NamedVarCall,
    var_name: &str,
) -> Option<ExprId> {
    supported_integral_derivative_presentation_route_for_integrate_call(
        ctx,
        integrate_call,
        var_name,
    )
    .into_presentation_target()
}

fn supported_integral_derivative_presentation_route_for_integrate_call(
    ctx: &mut Context,
    integrate_call: &NamedVarCall,
    var_name: &str,
) -> SupportedIntegralDerivativePresentationRoute {
    if integrate_call.var_name != var_name {
        return SupportedIntegralDerivativePresentationRoute::NoMatch;
    }

    let mut pending_non_success_route = SupportedIntegralDerivativePresentationRoute::NoMatch;

    // Verified source-return routes.
    if let Some(route) =
        SupportedIntegralDerivativePresentationRoute::observe_initial_verified_source_route(
            &mut pending_non_success_route,
            initial_verified_source_integral_derivative_route(
                ctx,
                integrate_call.target,
                &integrate_call.var_name,
            ),
        )
    {
        return route;
    }

    // Source-side presentation exception.
    if let Some(route) =
        SupportedIntegralDerivativePresentationRoute::observe_arctan_polynomial_source_target(
            arctan_polynomial_source_integral_derivative_shortcut(
                ctx,
                integrate_call.target,
                &integrate_call.var_name,
            ),
        )
    {
        return route;
    }

    if cas_math::symbolic_integration_support::integrate_symbolic_is_positive_rational_quadratic_arctan_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return SupportedIntegralDerivativePresentationRoute::ReciprocalPositiveQuadraticArctanSource(
            integrate_call.target,
        );
    }

    // Verified rational/substitution source-return routes.
    if let Some(route) =
        SupportedIntegralDerivativePresentationRoute::observe_rational_substitution_route(
            &mut pending_non_success_route,
            rational_substitution_verified_integral_derivative_route(
                ctx,
                integrate_call.target,
                &integrate_call.var_name,
            ),
        )
    {
        return route;
    }

    // Held-presentation routes that keep condition collection at the caller.
    if let Some(route) =
        SupportedIntegralDerivativePresentationRoute::observe_held_presentation_route(
            &mut pending_non_success_route,
            held_presentation_integral_derivative_route(
                ctx,
                integrate_call.target,
                &integrate_call.var_name,
            ),
        )
    {
        return route;
    }

    // Late verified source-return routes.
    if let Some(route) = SupportedIntegralDerivativePresentationRoute::observe_power_inverse_route(
        &mut pending_non_success_route,
        power_inverse_verified_integral_derivative_route(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ),
    ) {
        return route;
    }

    let final_route = final_presentation_integral_derivative_route(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    );
    SupportedIntegralDerivativePresentationRoute::complete_with_final_presentation_route(
        pending_non_success_route,
        final_route,
    )
}

pub(super) fn supported_integral_diff_shortcut_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let integrate_call = try_extract_integrate_call(ctx, target)?;
    let compact = supported_integral_derivative_presentation_for_integrate_call(
        ctx,
        &integrate_call,
        var_name,
    )?;
    Some(conditioned_integral_derivative_shortcut_result(
        ctx,
        compact,
        &integrate_call,
    ))
}

pub(super) fn supported_integral_diff_shortcut_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) =
        supported_integral_diff_shortcut_presentation(ctx, target, &call.var_name)?;
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
    use super::*;
    use cas_ast::hold;
    use cas_parser::parse;

    #[test]
    fn derivative_presentation_accepts_default_integrate_variable() {
        let mut ctx = Context::new();
        let target = parse("integrate(sin(2*x+1))", &mut ctx).unwrap();

        assert!(supported_integral_derivative_presentation(&mut ctx, target, "x").is_some());
    }

    #[test]
    fn derivative_presentation_rejects_integrate_variable_mismatch() {
        let mut ctx = Context::new();
        let target = parse("integrate(sin(2*x+1), y)", &mut ctx).unwrap();

        assert_eq!(
            supported_integral_derivative_presentation(&mut ctx, target, "x"),
            None
        );
    }

    #[test]
    fn derivative_presentation_route_preserves_final_verification_failure_signal() {
        let mut ctx = Context::new();
        let target = parse("integrate(sin(2*x+1)+exp(x^2), x)", &mut ctx).unwrap();
        let integrate_call = try_extract_integrate_call(&ctx, target).unwrap();

        assert!(matches!(
            supported_integral_derivative_presentation_route_for_integrate_call(
                &mut ctx,
                &integrate_call,
                "x"
            ),
            SupportedIntegralDerivativePresentationRoute::FinalPresentationSourceDirectTrigAffineVerificationFailed
        ));
        assert_eq!(
            supported_integral_derivative_presentation(&mut ctx, target, "x"),
            None
        );
    }

    #[test]
    fn conditioned_diff_shortcut_wraps_compact_and_preserves_condition_collection() {
        let mut ctx = Context::new();
        let target = parse("integrate(sin(2*x+1), x)", &mut ctx).unwrap();

        let (held, conditions) =
            supported_integral_diff_shortcut_presentation(&mut ctx, target, "x").unwrap();

        assert!(hold::is_hold(&ctx, held));
        assert!(conditions.is_empty());
    }

    #[test]
    fn diff_shortcut_rewrite_finalizes_compact_integral_source() {
        let mut ctx = Context::new();
        let target = parse("integrate(sin(2*x+1), x)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite = supported_integral_diff_shortcut_rewrite(&mut ctx, &call, target).unwrap();

        assert!(hold::is_hold(&ctx, rewrite.new_expr));
        assert!(rewrite.required_conditions.is_empty());
    }

    #[test]
    fn diff_shortcut_returns_positive_rational_quadratic_arctan_source() {
        let mut ctx = Context::new();
        let target = parse("integrate(1/((a*x+b)^2+2), x)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite = supported_integral_diff_shortcut_rewrite(&mut ctx, &call, target).unwrap();
        let compact = cas_ast::hold::unwrap_internal_hold(&ctx, rewrite.new_expr);

        assert_eq!(compact, parse("1/((a*x+b)^2+2)", &mut ctx).unwrap());
        let required_displays: Vec<_> = rewrite
            .required_conditions
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();
        assert!(
            !required_displays.iter().any(|display| display == "a ≠ 0"),
            "source shortcut must not inherit primitive-only slope condition: {required_displays:?}"
        );
    }
}

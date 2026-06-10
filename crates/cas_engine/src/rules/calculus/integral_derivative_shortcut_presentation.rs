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
//! 8. reciprocal trig derivative product: verified source return.
//! 9. rational linear partial fraction: verified source return.
//! 10. rational linear positive quadratic: verified source return.
//! 11. fractional denominator power substitution: verified source return.
//! 12. sqrt derivative substitution: verified source return.
//! 13. arcsin inverse-sqrt product: source-side held compact/source return.
//! 14. affine/acosh inverse-sqrt product: verified held compact/source return.
//! 15. arcsin/asinh polynomial substitution: verified held source return.
//! 16. positive quadratic powers: verified source return.
//! 17. inverse hyperbolic sqrt reciprocal: verified source return.
//! 18. direct sqrt-trig logarithm compaction: verified compact return.
//! 19. sqrt-trig logarithm presentation: verified compact-only return.
//! 20. sqrt reciprocal trig product: source-side presentation return.
//! 21. direct trig affine argument: verified source return.
//!
//! Keep this map synchronized with the route order. A later extraction should
//! move a whole policy group at a time and preserve whether the returned
//! integrand was independently verified, held for presentation, or accepted as
//! a source-side presentation shortcut.
//!
//! Policy groups:
//! - Routes 1-5, 8-12, and 16-17 are verified source-return routes.
//! - Routes 6, 7, and 20 are source-side presentation exceptions.
//! - Routes 13-15 are held-presentation routes that preserve source or compact
//!   integrands for caller condition collection.
//! - Routes 18-19 are compact-only sqrt-trig-log presentation routes.
//! - Route 21 is the final verified direct-trig fallback.
//!
//! The groups are descriptive, not a registry. Source-order priority remains in
//! the function body.

use super::diff_rule_support::{
    diff_rewrite_with_conditions, finalize_diff_rewrite_with_conditions,
};
use super::integral_derivative_arctan_polynomial_routes::arctan_polynomial_source_integral_derivative_shortcut;
use super::integral_derivative_conditioned_return::conditioned_integral_derivative_shortcut_result;
use super::integral_derivative_final_presentation_routes::final_presentation_integral_derivative_route;
use super::integral_derivative_held_presentation_routes::held_presentation_integral_derivative_route;
use super::integral_derivative_presentation_route::SupportedIntegralDerivativePresentationRoute;
use super::integral_derivative_verified_power_inverse_routes::power_inverse_verified_integral_derivative_route;
use super::integral_derivative_verified_rational_substitution_routes::rational_substitution_verified_integral_derivative_route;
use super::integral_derivative_verified_source_routes::initial_verified_source_integral_derivative_route;
use super::integration::{integrate_with_trace, IntegrationTraceKind};
use super::reciprocal_trig_derivative_product_source::{
    verified_affine_reciprocal_trig_derivative_product_source_with_domain,
    verified_reciprocal_trig_derivative_product_source_with_domain,
};
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{try_extract_integrate_call, NamedVarCall};
use cas_ast::{ConditionPredicate, Context, ExprId};

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

    if let Some((source_target, _required_nonzero)) =
        verified_reciprocal_trig_derivative_product_source_with_domain(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
    {
        return SupportedIntegralDerivativePresentationRoute::InitialVerifiedSource(source_target);
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
    if integrate_call.var_name == var_name {
        if let Some((compact, required_nonzero)) =
            verified_reciprocal_trig_derivative_product_source_with_domain(
                ctx,
                integrate_call.target,
                &integrate_call.var_name,
            )
        {
            return Some((
                cas_ast::hold::wrap_hold(ctx, compact),
                vec![crate::ImplicitCondition::NonZero(required_nonzero)],
            ));
        }

        if let Some((source, required_nonzero)) =
            verified_affine_reciprocal_trig_derivative_product_source_with_domain(
                ctx,
                integrate_call.target,
                &integrate_call.var_name,
            )
        {
            return Some((
                cas_ast::hold::wrap_hold(ctx, source),
                vec![crate::ImplicitCondition::NonZero(required_nonzero)],
            ));
        }
    }

    if let Some((source, required_conditions)) = algorithmic_backend_integral_source_with_conditions(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
        var_name,
    ) {
        return Some((source, required_conditions));
    }

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

pub(super) fn algorithmic_backend_integral_source_with_conditions(
    ctx: &mut Context,
    target: ExprId,
    integrate_var_name: &str,
    diff_var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if integrate_var_name != diff_var_name {
        return None;
    }

    let outcome = integrate_with_trace(ctx, target, integrate_var_name)?;
    if outcome.trace_kind != IntegrationTraceKind::AlgorithmicBackendSummary {
        return None;
    }

    Some((
        target,
        outcome
            .required_conditions
            .into_iter()
            .filter_map(backend_condition_to_implicit_condition)
            .collect(),
    ))
}

fn backend_condition_to_implicit_condition(
    condition: ConditionPredicate,
) -> Option<crate::ImplicitCondition> {
    match condition {
        ConditionPredicate::NonZero(expr) => Some(crate::ImplicitCondition::NonZero(expr)),
        ConditionPredicate::Positive(expr) => Some(crate::ImplicitCondition::Positive(expr)),
        ConditionPredicate::NonNegative(expr) => Some(crate::ImplicitCondition::NonNegative(expr)),
        ConditionPredicate::LowerBound { expr, lower } => {
            Some(crate::ImplicitCondition::LowerBound(expr, lower))
        }
        ConditionPredicate::Defined(_)
        | ConditionPredicate::InvTrigPrincipalRange { .. }
        | ConditionPredicate::EqZero(_)
        | ConditionPredicate::EqOne(_) => None,
    }
}

pub(super) fn reciprocal_trig_derivative_product_integral_diff_shortcut_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let integrate_call = try_extract_integrate_call(ctx, target)?;
    if integrate_call.var_name != call.var_name {
        return None;
    }

    let (compact, required_nonzero) =
        verified_reciprocal_trig_derivative_product_source_with_domain(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )?;
    let held = cas_ast::hold::wrap_hold(ctx, compact);
    Some(diff_rewrite_with_conditions(
        ctx,
        call,
        held,
        vec![crate::ImplicitCondition::NonZero(required_nonzero)],
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
    fn diff_shortcut_returns_algorithmic_backend_affine_quotient_source() {
        let mut ctx = Context::new();
        let target = parse("integrate((3*x+c)/(2*x+b), x)", &mut ctx).unwrap();
        let integrate_call = try_extract_integrate_call(&ctx, target).unwrap();

        let (result, required_conditions) =
            supported_integral_diff_shortcut_presentation(&mut ctx, target, "x")
                .expect("expected verified algorithmic backend shortcut");

        assert_eq!(result, integrate_call.target);
        let required_displays: Vec<_> = required_conditions
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();
        assert_eq!(required_displays, vec!["b + 2 * x ≠ 0"]);
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

    #[test]
    fn diff_shortcut_returns_verified_cosecant_cotangent_source() {
        let mut ctx = Context::new();
        let target = parse("integrate(k*a*csc(a*x+b)*cot(a*x+b), x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            supported_integral_diff_shortcut_presentation(&mut ctx, target, "x")
                .expect("expected csc*cot derivative-product shortcut");

        assert_eq!(
            hold::unwrap_internal_hold(&ctx, result),
            try_extract_integrate_call(&ctx, target).unwrap().target
        );
        assert_eq!(required_conditions.len(), 1);
    }

    #[test]
    fn diff_shortcut_returns_verified_negative_secant_tangent_source() {
        let mut ctx = Context::new();
        let target = parse("integrate(-k*a*sec(b-a*x)*tan(b-a*x), x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            supported_integral_diff_shortcut_presentation(&mut ctx, target, "x")
                .expect("expected sec*tan derivative-product shortcut");

        assert_eq!(
            hold::unwrap_internal_hold(&ctx, result),
            try_extract_integrate_call(&ctx, target).unwrap().target
        );
        let required_displays: Vec<_> = required_conditions
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();
        assert_eq!(required_displays, vec!["cos(b - a * x) ≠ 0"]);
    }

    #[test]
    fn diff_shortcut_returns_verified_affine_reciprocal_trig_source_without_explicit_du_factor() {
        let mut ctx = Context::new();

        for (raw, expected_condition) in [
            ("integrate(sec(2*x+1)*tan(2*x+1), x)", "cos(2 * x + 1) ≠ 0"),
            ("integrate(csc(2*x+1)*cot(2*x+1), x)", "sin(2 * x + 1) ≠ 0"),
            ("integrate(sec(1-2*x)*tan(1-2*x), x)", "cos(1 - 2 * x) ≠ 0"),
            ("integrate(csc(1-2*x)*cot(1-2*x), x)", "sin(1 - 2 * x) ≠ 0"),
        ] {
            let target = parse(raw, &mut ctx).unwrap();
            let (result, required_conditions) =
                supported_integral_diff_shortcut_presentation(&mut ctx, target, "x")
                    .unwrap_or_else(|| panic!("expected affine reciprocal trig shortcut: {raw}"));

            assert_eq!(
                hold::unwrap_internal_hold(&ctx, result),
                try_extract_integrate_call(&ctx, target).unwrap().target,
                "{raw}"
            );
            let required_displays: Vec<_> = required_conditions
                .iter()
                .map(|condition| condition.display(&ctx))
                .collect();
            assert_eq!(required_displays, vec![expected_condition], "{raw}");
        }
    }

    #[test]
    fn diff_shortcut_rejects_reciprocal_trig_product_without_derivative_factor() {
        let mut ctx = Context::new();

        for raw in [
            "integrate(sec(x^2)*tan(x^2), x)",
            "integrate(csc(x^2)*cot(x^2), x)",
        ] {
            let target = parse(raw, &mut ctx).unwrap();
            assert!(
                supported_integral_diff_shortcut_presentation(&mut ctx, target, "x").is_none(),
                "{raw}"
            );
        }
    }
}

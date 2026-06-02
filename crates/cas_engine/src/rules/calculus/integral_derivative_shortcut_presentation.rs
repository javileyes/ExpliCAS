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
//! 7. rational linear partial fraction: verified source return.
//! 8. rational linear positive quadratic: verified source return.
//! 9. fractional denominator power substitution: verified source return.
//! 10. sqrt derivative substitution: verified source return.
//! 11. arcsin inverse-sqrt product: source-side held compact/source return.
//! 12. affine/acosh inverse-sqrt product: verified held compact/source return.
//! 13. arcsin/asinh polynomial substitution: verified held source return.
//! 14. positive quadratic powers: verified source return.
//! 15. inverse hyperbolic sqrt reciprocal: verified source return.
//! 16. direct sqrt-trig logarithm compaction: verified compact return.
//! 17. sqrt-trig logarithm presentation: verified compact-only return.
//! 18. sqrt reciprocal trig product: source-side presentation return.
//! 19. direct trig affine argument: verified source return.
//!
//! Keep this map synchronized with the route order. A later extraction should
//! move a whole policy group at a time and preserve whether the returned
//! integrand was independently verified, held for presentation, or accepted as
//! a source-side presentation shortcut.
//!
//! Policy groups:
//! - Routes 1-5, 7-10, and 14-15 are verified source-return routes.
//! - Routes 6 and 18 are source-side presentation exceptions.
//! - Routes 11-13 are held-presentation routes that preserve source or compact
//!   integrands for caller condition collection.
//! - Routes 16-17 are compact-only sqrt-trig-log presentation routes.
//! - Route 19 is the final verified direct-trig fallback.
//!
//! The groups are descriptive, not a registry. Source-order priority remains in
//! the function body.

use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::integral_derivative_arctan_polynomial_routes::arctan_polynomial_source_integral_derivative_shortcut;
use super::integral_derivative_conditioned_return::conditioned_integral_derivative_shortcut_result;
use super::integral_derivative_final_presentation_routes::final_presentation_integral_derivative_shortcut;
use super::integral_derivative_held_presentation_routes::held_presentation_integral_derivative_shortcut;
use super::integral_derivative_verified_power_inverse_routes::power_inverse_verified_integral_derivative_shortcut;
use super::integral_derivative_verified_rational_substitution_routes::rational_substitution_verified_integral_derivative_shortcut;
use super::integral_derivative_verified_source_routes::initial_verified_source_integral_derivative_shortcut;
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
    if integrate_call.var_name != var_name {
        return None;
    }

    // Verified source-return routes.
    if let Some(source_target) = initial_verified_source_integral_derivative_shortcut(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return Some(source_target);
    }

    // Source-side presentation exception.
    if let Some(source_target) = arctan_polynomial_source_integral_derivative_shortcut(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return Some(source_target);
    }

    // Verified rational/substitution source-return routes.
    if let Some(source_target) = rational_substitution_verified_integral_derivative_shortcut(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return Some(source_target);
    }

    // Held-presentation routes that keep condition collection at the caller.
    if let Some(held_target) = held_presentation_integral_derivative_shortcut(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return Some(held_target);
    }

    // Late verified source-return routes.
    if let Some(source_target) = power_inverse_verified_integral_derivative_shortcut(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return Some(source_target);
    }

    // Final compact/source presentation routes and verified fallback.
    if let Some(final_target) = final_presentation_integral_derivative_shortcut(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return Some(final_target);
    }

    None
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
}

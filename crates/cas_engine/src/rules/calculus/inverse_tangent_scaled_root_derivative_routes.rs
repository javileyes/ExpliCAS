//! Direct diff routes for inverse-tangent scaled-root derivative forms.
//!
//! This module preserves the local `diff_rule` priority for scaled sqrt
//! polynomial and symbolic-denominator shortcuts. It intentionally stops before
//! the positive-shift arctangent family, whose presentation policy is separate.

use super::diff_rule_support::{
    diff_rewrite_with_conditions, finalize_diff_rewrite_with_conditions,
};
use super::inverse_tangent_scaled_root_derivative_presentation::{
    atanh_sqrt_over_symbolic_constant_derivative_shortcut,
    constant_scaled_inverse_tangent_linear_positive_rational_radius_derivative_shortcut,
    constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut,
    inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut,
    inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut,
};
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn inverse_tangent_scaled_root_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    if let Some((result, required_conditions)) =
        atanh_sqrt_over_symbolic_constant_derivative_shortcut(ctx, target, &call.var_name)
    {
        return Some(diff_rewrite_with_conditions(
            ctx,
            call,
            result,
            required_conditions,
        ));
    }

    let (result, required_conditions) =
        inverse_tangent_scaled_root_derivative_route(ctx, target, &call.var_name)?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        required_conditions,
    ))
}

pub(super) fn inverse_tangent_scaled_root_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    constant_scaled_inverse_tangent_linear_positive_rational_radius_derivative_shortcut(
        ctx, target, var_name,
    )
    .or_else(|| inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut(ctx, target, var_name))
    .or_else(|| {
        inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(ctx, target, var_name)
    })
    .or_else(|| atanh_sqrt_over_symbolic_constant_derivative_shortcut(ctx, target, var_name))
    .or_else(|| {
        constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
            ctx, target, var_name,
        )
    })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::presentation_utils::unwrap_internal_hold_for_calculus;
    use super::inverse_tangent_scaled_root_derivative_route;
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn inverse_tangent_scaled_root_rewrite_preserves_finalized_conditions() {
        let mut ctx = Context::new();
        let target = parse("arctan(2*sqrt(x+1))", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite =
            super::inverse_tangent_scaled_root_derivative_rewrite(&mut ctx, &call, target).unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, rewrite.new_expr);

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (sqrt(x + 1) * (4 * x + 5))"
        );
        let rendered_conditions: Vec<_> = rewrite
            .required_conditions
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();
        assert_eq!(rendered_conditions, vec!["x > -1"]);
    }

    #[test]
    fn inverse_tangent_scaled_root_route_preserves_scaled_polynomial_conditions() {
        let mut ctx = Context::new();
        let target = parse("arctan(2*sqrt(x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            inverse_tangent_scaled_root_derivative_route(&mut ctx, target, "x").unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (sqrt(x + 1) * (4 * x + 5))"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > -1");
    }

    #[test]
    fn inverse_tangent_scaled_root_route_preserves_symbolic_denominator_conditions() {
        let mut ctx = Context::new();
        let target = parse("atanh(2*(sqrt(x+1)/a))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            inverse_tangent_scaled_root_derivative_route(&mut ctx, target, "x").unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "a / (sqrt(x + 1) * (a^2 - 4 * x - 4))"
        );
        assert_eq!(required_conditions.len(), 3);
        assert_eq!(required_conditions[0].display(&ctx), "x > -1");
        assert!(matches!(
            required_conditions[1],
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "a"
        ));
        assert!(matches!(
            required_conditions[2],
            crate::ImplicitCondition::Positive(required)
                if rendered(&ctx, required) == "a^2 - 4 * x - 4"
        ));
    }

    #[test]
    fn inverse_tangent_scaled_root_rewrite_handles_linear_positive_rational_radius() {
        let mut ctx = Context::new();
        let target = parse("arctan(sqrt(2)*(a*x+b)/2)/(sqrt(2)*a)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite =
            super::inverse_tangent_scaled_root_derivative_rewrite(&mut ctx, &call, target).unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, rewrite.new_expr);

        assert_eq!(rendered(&ctx, derivative), "1 / ((a * x + b)^2 + 2)");
        let rendered_conditions: Vec<_> = rewrite
            .required_conditions
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();
        assert_eq!(rendered_conditions, vec!["a ≠ 0"]);
    }
}

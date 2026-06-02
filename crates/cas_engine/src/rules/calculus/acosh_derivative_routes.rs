//! Family-owned routing for direct `acosh` derivative presentations.
//!
//! This keeps the `diff` rule focused on priority orchestration while this
//! module owns the `acosh` route order and required-condition propagation.

use super::acosh_affine_derivative_presentation::{
    acosh_affine_derivative_presentation, constant_scaled_acosh_affine_derivative_presentation,
};
use super::acosh_over_sqrt_derivative_presentation::{
    acosh_polynomial_over_sqrt_derivative_presentation,
    constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation,
};
use super::acosh_sqrt_derivative_presentation::acosh_sqrt_shifted_quadratic_derivative_presentation;
use super::acosh_strictly_positive_polynomial_derivative_presentation::acosh_strictly_positive_polynomial_derivative_presentation;
use super::diff_required_conditions::acosh_sqrt_diff_required_conditions;
use crate::ImplicitCondition;
use cas_ast::{Context, ExprId};

pub(super) fn acosh_direct_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<ImplicitCondition>)> {
    if let Some(result) =
        acosh_sqrt_shifted_quadratic_derivative_presentation(ctx, target, var_name)
    {
        return Some((result, acosh_sqrt_diff_required_conditions(ctx, target)));
    }
    if let Some((result, required_conditions)) =
        acosh_affine_derivative_presentation(ctx, target, var_name)
    {
        return Some((result, required_conditions));
    }
    if let Some((result, required_conditions)) =
        acosh_strictly_positive_polynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some((result, required_conditions));
    }

    acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, var_name)
}

pub(super) fn constant_scaled_acosh_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<ImplicitCondition>)> {
    constant_scaled_acosh_affine_derivative_presentation(ctx, target, var_name).or_else(|| {
        constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(ctx, target, var_name)
    })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{acosh_direct_derivative_route, constant_scaled_acosh_derivative_route};

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn acosh_direct_route_preserves_affine_domain_condition() {
        let mut ctx = Context::new();
        let target = parse("acosh(x+1)", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            acosh_direct_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "1 / (sqrt(x) * sqrt(x + 2))");
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 0");
    }

    #[test]
    fn constant_scaled_acosh_route_preserves_affine_domain_condition() {
        let mut ctx = Context::new();
        let target = parse("acosh(x+1)/2", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_acosh_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (2 * sqrt(x) * sqrt(x + 2))"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 0");
    }

    #[test]
    fn constant_scaled_acosh_route_preserves_polynomial_over_sqrt_domain_conditions() {
        let mut ctx = Context::new();
        let target = parse("2*acosh((x^2+x)/sqrt(5))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_acosh_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "2 * (2 * x + 1) / sqrt((x^2 + x)^2 - 5)"
        );
        assert_eq!(required_conditions.len(), 2);
        assert_eq!(required_conditions[0].display(&ctx), "(x^2 + x)^2 - 5 > 0");
        assert_eq!(
            required_conditions[1].display(&ctx),
            "x^2 + x - sqrt(5) > 0"
        );
    }
}

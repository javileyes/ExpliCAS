//! Final presentation routes for integration-backed derivative shortcuts.
//!
//! This module owns routes 16-19 from `integral_derivative_shortcut_presentation`.
//! It preserves the parent gate's final route order, including the early
//! `None` propagation from the sqrt-trig-log compact verifier.

use super::integral_derivative_direct_trig_affine_routes::direct_trig_affine_integral_derivative_shortcut;
use super::integral_derivative_sqrt_reciprocal_trig_routes::sqrt_reciprocal_trig_product_integral_derivative_shortcut;
use super::integral_derivative_sqrt_trig_log_routes::{
    sqrt_trig_log_integral_derivative_shortcut, SqrtTrigLogDerivativeRoute,
};
use cas_ast::{Context, ExprId};

pub(super) fn final_presentation_integral_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    match sqrt_trig_log_integral_derivative_shortcut(ctx, target, var_name)? {
        SqrtTrigLogDerivativeRoute::VerifiedCompact(compact) => return Some(compact),
        SqrtTrigLogDerivativeRoute::NoMatch | SqrtTrigLogDerivativeRoute::VerifiedNoCompact => {}
    }

    if let Some(source_target) =
        sqrt_reciprocal_trig_product_integral_derivative_shortcut(ctx, target, var_name)
    {
        return Some(source_target);
    }

    if let Some(source_target) =
        direct_trig_affine_integral_derivative_shortcut(ctx, target, var_name)
    {
        return Some(source_target);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn returns_compact_sqrt_trig_log_route() {
        let mut ctx = Context::new();
        let target = parse("tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();
        let compact =
            final_presentation_integral_derivative_shortcut(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn returns_source_for_sqrt_reciprocal_trig_product_route() {
        let mut ctx = Context::new();
        let target = parse("sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();

        assert_eq!(
            final_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn returns_source_for_direct_trig_affine_fallback() {
        let mut ctx = Context::new();
        let target = parse("sin(2*x+1)", &mut ctx).unwrap();

        assert_eq!(
            final_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            Some(target)
        );
    }

    #[test]
    fn rejects_expression_outside_final_routes() {
        let mut ctx = Context::new();
        let target = parse("exp(x^2)", &mut ctx).unwrap();

        assert_eq!(
            final_presentation_integral_derivative_shortcut(&mut ctx, target, "x"),
            None
        );
    }
}

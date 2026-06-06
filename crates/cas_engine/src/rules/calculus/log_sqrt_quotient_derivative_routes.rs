//! Direct diff route for log/square-root quotient derivative presentation.
//!
//! This route preserves the current `diff_rule` priority for the two reciprocal
//! orientations while the presentation module keeps owning result construction.

use super::log_sqrt_quotient_derivative_presentation::{
    log_over_sqrt_polynomial_derivative_presentation,
    sqrt_over_log_polynomial_derivative_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) fn log_sqrt_quotient_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    log_over_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
        .or_else(|| sqrt_over_log_polynomial_derivative_presentation(ctx, target, var_name))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::log_sqrt_quotient_derivative_presentation::{
        log_over_sqrt_polynomial_derivative_presentation,
        sqrt_over_log_polynomial_derivative_presentation,
    };
    use super::log_sqrt_quotient_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn log_sqrt_quotient_route_preserves_log_over_sqrt_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("ln(x+1)/sqrt(x+1)", &mut route_ctx).unwrap();
        let route_result =
            log_sqrt_quotient_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("ln(x+1)/sqrt(x+1)", &mut direct_ctx).unwrap();
        let direct_result =
            log_over_sqrt_polynomial_derivative_presentation(&mut direct_ctx, direct_target, "x")
                .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
    }

    #[test]
    fn log_sqrt_quotient_route_preserves_sqrt_over_log_hold() {
        let mut route_ctx = Context::new();
        let route_target = parse("sqrt(x)/(a*ln(x))", &mut route_ctx).unwrap();
        let route_result =
            log_sqrt_quotient_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("sqrt(x)/(a*ln(x))", &mut direct_ctx).unwrap();
        let direct_result =
            sqrt_over_log_polynomial_derivative_presentation(&mut direct_ctx, direct_target, "x")
                .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
    }

    #[test]
    fn log_sqrt_quotient_route_handles_unscaled_sqrt_over_log() {
        let mut ctx = Context::new();
        let target = parse("sqrt(x)/ln(x)", &mut ctx).unwrap();
        let result = log_sqrt_quotient_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, result),
            "(ln(x) - 2) / (2 * ln(x)^2 * sqrt(x))"
        );
    }
}

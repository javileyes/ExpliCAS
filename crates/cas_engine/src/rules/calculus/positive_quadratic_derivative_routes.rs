//! Direct diff routes for compact strictly-positive-quadratic results.
//!
//! This route preserves the local `diff_rule` priority for square-reduction
//! results before quotient results. The presentation helpers still own formula
//! construction and positivity recognition.

use super::positive_quadratic_presentation::{
    positive_quadratic_quotient_derivative_presentation,
    positive_quadratic_square_derivative_result_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) fn positive_quadratic_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    positive_quadratic_square_derivative_result_presentation(ctx, target, var_name)
        .or_else(|| positive_quadratic_quotient_derivative_presentation(ctx, target, var_name))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::positive_quadratic_presentation::{
        positive_quadratic_quotient_derivative_presentation,
        positive_quadratic_square_derivative_result_presentation,
    };
    use super::positive_quadratic_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn positive_quadratic_route_preserves_square_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("1/2*arctan(x) + x/(2*(x^2+1))", &mut route_ctx).unwrap();
        let route_result =
            positive_quadratic_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("1/2*arctan(x) + x/(2*(x^2+1))", &mut direct_ctx).unwrap();
        let direct_result = positive_quadratic_square_derivative_result_presentation(
            &mut direct_ctx,
            direct_target,
            "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
    }

    #[test]
    fn positive_quadratic_route_preserves_quotient_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("arctan(2*x)/(4*x^2+1)", &mut route_ctx).unwrap();
        let route_result =
            positive_quadratic_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("arctan(2*x)/(4*x^2+1)", &mut direct_ctx).unwrap();
        let direct_result = positive_quadratic_quotient_derivative_presentation(
            &mut direct_ctx,
            direct_target,
            "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
    }
}

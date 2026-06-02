//! Direct diff routes for condition-carrying asinh sqrt reciprocal-polynomial forms.
//!
//! This mirrors the local priority in `diff_rule`: direct target first, then
//! externally scaled target. Keep this route separate from sibling inverse
//! hyperbolic families whose domain and call-order policies are not adjacent.

use super::asinh_sqrt_constant_over_polynomial_presentation::{
    asinh_sqrt_constant_over_polynomial_presentation,
    scaled_asinh_sqrt_constant_over_polynomial_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) fn conditioned_asinh_sqrt_constant_over_polynomial_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    asinh_sqrt_constant_over_polynomial_presentation(ctx, target, var_name)
        .or_else(|| scaled_asinh_sqrt_constant_over_polynomial_presentation(ctx, target, var_name))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::conditioned_asinh_sqrt_constant_over_polynomial_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn conditioned_asinh_sqrt_route_preserves_scaled_route_result_and_condition() {
        let mut ctx = Context::new();
        let target = parse("2*asinh(sqrt(3/x))", &mut ctx).unwrap();
        let (derivative, required_condition) =
            conditioned_asinh_sqrt_constant_over_polynomial_derivative_route(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-2 * sqrt(3)/(2 * x * sqrt(x + 3))"
        );
        assert_eq!(required_condition.display(&ctx), "x > 0");
    }
}

//! Direct diff routes for inverse-tangent reciprocal-sqrt derivative forms.
//!
//! This module preserves the local `diff_rule` priority for this condition
//! vector family. It intentionally does not merge bounded inverse-trig or
//! inverse-hyperbolic routes, whose domain policies differ.

use super::inverse_tangent_polynomial_root_derivative_presentation::negative_arccot_sqrt_polynomial_derivative_shortcut;
use super::inverse_tangent_reciprocal_sqrt_derivative_presentation::{
    arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut,
    inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut,
};
use super::reciprocal_sqrt_product_derivative_presentation::inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation;
use cas_ast::{Context, ExprId};

pub(super) fn inverse_tangent_reciprocal_sqrt_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut(ctx, target, var_name)
        .or_else(|| negative_arccot_sqrt_polynomial_derivative_shortcut(ctx, target, var_name))
        .or_else(|| {
            inverse_tangent_reciprocal_sqrt_polynomial_product_derivative_presentation(
                ctx, target, var_name,
            )
        })
        .or_else(|| {
            inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut(ctx, target, var_name)
        })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::inverse_tangent_reciprocal_sqrt_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn inverse_tangent_reciprocal_sqrt_route_preserves_basic_result_and_conditions() {
        let mut ctx = Context::new();
        let target = parse("arctan(1/sqrt(x^2+x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            inverse_tangent_reciprocal_sqrt_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(2 * x + 1) / (2 * sqrt(x^2 + x + 1) * (x^2 + x + 2))"
        );
        assert!(required_conditions.is_empty());
    }
}

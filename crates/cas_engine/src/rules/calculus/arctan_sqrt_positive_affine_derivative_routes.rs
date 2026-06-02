//! Direct diff routes for arctangent sqrt-over-positive-affine forms.
//!
//! This route owns the local priority between the constant-scaled form and the
//! base positive-affine quotient form. It deliberately does not merge the
//! nearby positive-shift arctangent presentation, whose result compaction policy
//! is separate.

use super::arctan_sqrt_quotient_derivative_presentation::{
    arctan_sqrt_variable_over_positive_affine_derivative_presentation,
    constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) fn arctan_sqrt_positive_affine_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    constant_scaled_arctan_sqrt_variable_over_positive_affine_derivative_presentation(
        ctx, target, var_name,
    )
    .or_else(|| {
        arctan_sqrt_variable_over_positive_affine_derivative_presentation(ctx, target, var_name)
    })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::presentation_utils::unwrap_internal_hold_for_calculus;
    use super::arctan_sqrt_positive_affine_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn arctan_sqrt_positive_affine_route_preserves_base_conditions() {
        let mut ctx = Context::new();
        let target = parse("arctan(sqrt(x)/(x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            arctan_sqrt_positive_affine_derivative_route(&mut ctx, target, "x").unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "(1 - x) / (2 * sqrt(x) * (x^2 + 3 * x + 1))"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 0");
    }

    #[test]
    fn arctan_sqrt_positive_affine_route_preserves_constant_scale_conditions() {
        let mut ctx = Context::new();
        let target = parse("3*arctan(sqrt(x)/(x+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            arctan_sqrt_positive_affine_derivative_route(&mut ctx, target, "x").unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "3 * (1 - x) / (2 * (x^2 + 3 * x + 1) * sqrt(x))"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 0");
    }
}

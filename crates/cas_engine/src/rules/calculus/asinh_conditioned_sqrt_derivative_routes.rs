//! Direct diff routes for condition-carrying asinh sqrt reciprocal-polynomial forms.
//!
//! This mirrors the local priority in `diff_rule`: direct target first, then
//! externally scaled target. Keep this route separate from sibling inverse
//! hyperbolic families whose domain and call-order policies are not adjacent.

use super::asinh_sqrt_constant_over_polynomial_presentation::{
    asinh_sqrt_constant_over_polynomial_presentation,
    scaled_asinh_sqrt_constant_over_polynomial_presentation,
};
use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn conditioned_asinh_sqrt_constant_over_polynomial_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    asinh_sqrt_constant_over_polynomial_presentation(ctx, target, var_name)
        .or_else(|| scaled_asinh_sqrt_constant_over_polynomial_presentation(ctx, target, var_name))
}

pub(super) fn conditioned_asinh_sqrt_constant_over_polynomial_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_condition) =
        conditioned_asinh_sqrt_constant_over_polynomial_derivative_route(
            ctx,
            target,
            &call.var_name,
        )?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        std::iter::once(required_condition),
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{
        conditioned_asinh_sqrt_constant_over_polynomial_derivative_rewrite,
        conditioned_asinh_sqrt_constant_over_polynomial_derivative_route,
    };
    use crate::symbolic_calculus_call_support::NamedVarCall;

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

    #[test]
    fn conditioned_asinh_sqrt_rewrite_preserves_scaled_route_result_and_condition() {
        let mut ctx = Context::new();
        let target = parse("2*asinh(sqrt(3/x))", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = conditioned_asinh_sqrt_constant_over_polynomial_derivative_rewrite(
            &mut ctx, &call, target,
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "-2 * sqrt(3)/(2 * x * sqrt(x + 3))"
        );
        assert_eq!(rewrite.required_conditions.len(), 1);
        assert_eq!(rewrite.required_conditions[0].display(&ctx), "x > 0");
    }
}

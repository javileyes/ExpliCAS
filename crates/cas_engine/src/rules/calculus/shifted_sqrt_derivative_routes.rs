//! Direct diff routes for shifted-square-root quotient/product forms.
//!
//! This route preserves the local `diff_rule` priority across the compact
//! reciprocal, product-denominator, and sqrt-over-shifted-sqrt variants. The
//! presentation helpers still own formula construction and domain policy.

use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::shifted_sqrt_derivative_presentation::{
    reciprocal_positive_shifted_sqrt_derivative,
    reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative,
    sqrt_over_positive_shifted_sqrt_derivative,
};
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn shifted_sqrt_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) = shifted_sqrt_derivative_route(ctx, target, &call.var_name)?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        required_conditions,
    ))
}

pub(super) fn shifted_sqrt_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    reciprocal_positive_shifted_sqrt_derivative(ctx, target, var_name)
        .map(|result| (result, Vec::new()))
        .or_else(|| reciprocal_sqrt_times_nonzero_shifted_sqrt_derivative(ctx, target, var_name))
        .or_else(|| {
            let (result, required_positive) =
                sqrt_over_positive_shifted_sqrt_derivative(ctx, target, var_name)?;
            Some((
                result,
                vec![crate::ImplicitCondition::Positive(required_positive)],
            ))
        })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::presentation_utils::unwrap_internal_hold_for_calculus;
    use super::{shifted_sqrt_derivative_rewrite, shifted_sqrt_derivative_route};
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn shifted_sqrt_rewrite_preserves_finalized_conditions() {
        let mut ctx = Context::new();
        let target = parse("sqrt(x)/(sqrt(x)+1)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = shifted_sqrt_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "1 / (2 * sqrt(x) * (sqrt(x) + 1)^2)"
        );
        assert_eq!(rewrite.required_conditions.len(), 1);
        assert_eq!(rewrite.required_conditions[0].display(&ctx), "x > 0");
    }

    #[test]
    fn shifted_sqrt_route_preserves_reciprocal_result_without_extra_conditions() {
        let mut ctx = Context::new();
        let target = parse("1/(sqrt(x)+1)", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            shifted_sqrt_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-1 / (2 * sqrt(x) * (sqrt(x) + 1)^2)"
        );
        assert!(required_conditions.is_empty());
    }

    #[test]
    fn shifted_sqrt_route_preserves_product_conditions() {
        let mut ctx = Context::new();
        let target = parse("1/(sqrt(x)*(sqrt(x)+1))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            shifted_sqrt_derivative_route(&mut ctx, target, "x").unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "-(2 * sqrt(x) + 1) / (2 * x * sqrt(x) * (sqrt(x) + 1)^2)"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 0");
    }

    #[test]
    fn shifted_sqrt_route_preserves_sqrt_over_shifted_conditions() {
        let mut ctx = Context::new();
        let target = parse("sqrt(x)/(sqrt(x)+1)", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            shifted_sqrt_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (2 * sqrt(x) * (sqrt(x) + 1)^2)"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x > 0");
    }
}

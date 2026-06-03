//! Early sqrt derivative routes kept near the `DiffRule` priority point.
//!
//! This is a route group for condition-bearing presentation shortcuts, not a
//! generic sqrt derivative registry.

use cas_ast::{Context, ExprId};

use crate::symbolic_calculus_call_support::NamedVarCall;
use crate::Rewrite;

use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::domain_checks::append_positive_required_conditions;
use super::reciprocal_sqrt_product_derivative_presentation::reciprocal_sqrt_polynomial_product_derivative_presentation;
use super::sqrt_additive_trig_derivative_presentation::sqrt_additive_trig_polynomial_derivative_presentation;

pub(super) fn sqrt_early_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let mut shortcut_required_conditions = Vec::new();
    let result = sqrt_additive_trig_polynomial_derivative_presentation(ctx, target, &call.var_name)
        .map(|(result, required_positive, required_conditions)| {
            append_positive_required_conditions(
                &mut shortcut_required_conditions,
                required_positive,
                required_conditions,
            );
            result
        })
        .or_else(|| {
            reciprocal_sqrt_polynomial_product_derivative_presentation(ctx, target, &call.var_name)
        })?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        shortcut_required_conditions,
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use crate::symbolic_calculus_call_support::NamedVarCall;

    use super::super::domain_checks::append_positive_required_conditions;
    use super::super::sqrt_additive_trig_derivative_presentation::sqrt_additive_trig_polynomial_derivative_presentation;
    use super::sqrt_early_derivative_rewrite;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn sqrt_early_rewrite_preserves_additive_trig_result_and_conditions() {
        let mut ctx = Context::new();
        let target = parse("sqrt(sin(2*x)+cos(x)-2/x)", &mut ctx).unwrap();
        let (route_result, required_positive, route_conditions) =
            sqrt_additive_trig_polynomial_derivative_presentation(&mut ctx, target, "x").unwrap();
        let mut expected_conditions = Vec::new();
        append_positive_required_conditions(
            &mut expected_conditions,
            required_positive,
            route_conditions,
        );
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = sqrt_early_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, route_result)
        );
        assert_eq!(rewrite.required_conditions, expected_conditions);
    }
}

//! Direct diff route for sqrt of elementary-function radicands.
//!
//! The presentation helper owns the derivative shape; this route owns the
//! required positive radicand condition that `diff_rule` previously appended.

use super::sqrt_elementary_function_derivative_presentation::sqrt_elementary_function_derivative_presentation;
use cas_ast::{Context, ExprId};
use cas_math::root_forms::extract_square_root_base;

pub(super) fn sqrt_elementary_function_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let result = sqrt_elementary_function_derivative_presentation(ctx, target, var_name)?;
    let required_conditions = extract_square_root_base(ctx, target)
        .map(crate::ImplicitCondition::Positive)
        .into_iter()
        .collect();
    Some((result, required_conditions))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::sqrt_elementary_function_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn sqrt_elementary_route_preserves_result_and_positive_condition() {
        let mut ctx = Context::new();
        let target = parse("sqrt(sin(x))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            sqrt_elementary_function_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "cos(x) / (2 * sqrt(sin(x)))");
        let rendered_conditions = required_conditions
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect::<Vec<_>>();
        assert_eq!(rendered_conditions, vec!["sin(x) > 0"]);
    }
}

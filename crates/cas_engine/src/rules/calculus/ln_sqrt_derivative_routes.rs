//! Direct diff routes for `ln(sqrt(...))` derivative families.
//!
//! The presentation modules own each compact derivative shape. This route keeps
//! their priority and required-condition handling out of the top-level diff rule.

use super::ln_sqrt_polynomial_direct_derivative_presentation::ln_sqrt_plus_polynomial_direct_derivative_presentation;
use super::ln_sqrt_polynomial_gap_derivative_presentation::ln_sqrt_polynomial_gap_derivative_presentation;
use super::ln_sqrt_positive_shift_derivative_presentation::ln_sqrt_positive_shift_nonpolynomial_derivative_presentation;
use cas_ast::{Context, ExprId};

pub(super) fn ln_sqrt_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(ctx, target, var_name)
        .or_else(|| {
            ln_sqrt_polynomial_gap_derivative_presentation(ctx, target, var_name)
                .map(|result| (result, Vec::new()))
        })
        .or_else(|| ln_sqrt_plus_polynomial_direct_derivative_presentation(ctx, target, var_name))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::ln_sqrt_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn ln_sqrt_route_preserves_positive_shift_condition() {
        let mut ctx = Context::new();
        let target = parse("ln(1+sqrt(sin(x)+2))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            ln_sqrt_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "cos(x) / (2 * sqrt(sin(x) + 2) * (sqrt(sin(x) + 2) + 1))"
        );
        let rendered_conditions = required_conditions
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect::<Vec<_>>();
        assert_eq!(rendered_conditions, vec!["sin(x) + 2 > 0"]);
    }

    #[test]
    fn ln_sqrt_route_preserves_gap_route_without_conditions() {
        let mut ctx = Context::new();
        let target = parse("ln(sqrt(x^2+1)+x)", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            ln_sqrt_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "1 / sqrt(x^2 + 1)");
        assert!(required_conditions.is_empty());
    }
}

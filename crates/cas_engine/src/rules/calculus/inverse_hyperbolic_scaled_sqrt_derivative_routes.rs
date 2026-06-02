//! Direct diff routes for constant-scaled inverse-hyperbolic sqrt-polynomial forms.
//!
//! This intentionally mirrors the local priority from `diff_rule` and does not
//! call the broader inverse-hyperbolic root post-calculus coordinator.

use super::acosh_sqrt_derivative_presentation::constant_scaled_acosh_sqrt_polynomial_derivative_presentation;
use super::asinh_sqrt_derivative_presentation::constant_scaled_asinh_sqrt_polynomial_derivative_presentation;
use super::atanh_sqrt_derivative_presentation::constant_scaled_atanh_sqrt_polynomial_derivative_presentation;
use cas_ast::{Context, ExprId};

pub(super) fn constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    constant_scaled_asinh_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
        .or_else(|| {
            constant_scaled_atanh_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
        })
        .or_else(|| {
            constant_scaled_acosh_sqrt_polynomial_derivative_presentation(ctx, target, var_name)
        })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn constant_scaled_inverse_hyperbolic_sqrt_route_preserves_asinh_result() {
        let mut ctx = Context::new();
        let target = parse("asinh(sqrt(x))/2", &mut ctx).unwrap();
        let derivative = constant_scaled_inverse_hyperbolic_sqrt_polynomial_derivative_route(
            &mut ctx, target, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (4 * sqrt(x + 1) * sqrt(x))"
        );
    }
}

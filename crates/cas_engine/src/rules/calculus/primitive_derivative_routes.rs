//! Family-owned routing for compact primitive derivative presentations.
//!
//! These routes recognize derivatives of already-constructed calculus
//! primitives. They do not create required conditions; callers preserve the
//! surrounding `diff` condition collection policy.

use super::exponential_derivative_presentation::exp_trig_by_parts_primitive_derivative_presentation;
use super::hyperbolic_primitive_derivative_presentation::affine_hyperbolic_odd_primitive_derivative_presentation;
use super::tanh_primitive_derivative_presentation::{
    affine_tanh_even_primitive_derivative_presentation,
    tanh_cubic_sech_fourth_primitive_derivative_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) fn primitive_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    exp_trig_by_parts_primitive_derivative_presentation(ctx, target, var_name)
        .or_else(|| tanh_cubic_sech_fourth_primitive_derivative_presentation(ctx, target, var_name))
        .or_else(|| affine_tanh_even_primitive_derivative_presentation(ctx, target, var_name))
        .or_else(|| affine_hyperbolic_odd_primitive_derivative_presentation(ctx, target, var_name))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::primitive_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn primitive_route_preserves_exp_trig_by_parts_derivative() {
        let mut ctx = Context::new();
        let target = parse("1/4*exp(2*x+1)*(sin(2*x+1)-cos(2*x+1))", &mut ctx).unwrap();
        let derivative = primitive_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "sin(2 * x + 1) * e^(2 * x + 1)");
    }

    #[test]
    fn primitive_route_preserves_tanh_even_derivative() {
        let mut ctx = Context::new();
        let target = parse(
            "x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5)",
            &mut ctx,
        )
        .unwrap();
        let derivative = primitive_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "tanh(2 * x + 1)^6");
    }

    #[test]
    fn primitive_route_preserves_tanh_cubic_sech_fourth_derivative() {
        let mut ctx = Context::new();
        let target = parse("k*tanh(x^2+b)-k*tanh(x^2+b)^3/3", &mut ctx).unwrap();
        let derivative = primitive_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "k * 2 * x^(2 - 1) / cosh(x^2 + b)^4"
        );
    }

    #[test]
    fn primitive_route_preserves_hyperbolic_odd_derivative() {
        let mut ctx = Context::new();
        let target = parse("1/2*(sinh(2*x+1)+1/3*sinh(2*x+1)^3)", &mut ctx).unwrap();
        let derivative = primitive_derivative_route(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "cosh(2 * x + 1)^3");
    }
}

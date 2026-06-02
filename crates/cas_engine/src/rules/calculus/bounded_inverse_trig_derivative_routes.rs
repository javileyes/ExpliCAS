//! Direct diff routes for bounded inverse-trig argument forms.
//!
//! This route preserves the local `diff_rule` priority across compact surd
//! quotient, unit-interval affine, and self-normalized projection variants. The
//! presentation modules still own result construction and argument policy.

use super::bounded_inverse_trig_projection_presentation::bounded_inverse_trig_self_normalized_projection_derivative_presentation;
use super::inverse_trig_derivative_presentation::{
    bounded_inverse_trig_surd_quotient_compact_derivative,
    unit_interval_bounded_inverse_trig_derivative_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) fn bounded_inverse_trig_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, var_name)
        .or_else(|| {
            unit_interval_bounded_inverse_trig_derivative_presentation(ctx, target, var_name)
                .map(|compact| cas_ast::hold::wrap_hold(ctx, compact))
        })
        .or_else(|| {
            bounded_inverse_trig_self_normalized_projection_derivative_presentation(
                ctx, target, var_name,
            )
        })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::bounded_inverse_trig_projection_presentation::bounded_inverse_trig_self_normalized_projection_derivative_presentation;
    use super::super::inverse_trig_derivative_presentation::{
        bounded_inverse_trig_surd_quotient_compact_derivative,
        unit_interval_bounded_inverse_trig_derivative_presentation,
    };
    use super::bounded_inverse_trig_derivative_route;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn bounded_inverse_trig_route_preserves_surd_quotient_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("arcsin(x/sqrt(4))", &mut route_ctx).unwrap();
        let route_result =
            bounded_inverse_trig_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("arcsin(x/sqrt(4))", &mut direct_ctx).unwrap();
        let direct_result = bounded_inverse_trig_surd_quotient_compact_derivative(
            &mut direct_ctx,
            direct_target,
            "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
    }

    #[test]
    fn bounded_inverse_trig_route_preserves_unit_interval_hold() {
        let mut route_ctx = Context::new();
        let route_target = parse("arcsin(2*x-1)", &mut route_ctx).unwrap();
        let route_result =
            bounded_inverse_trig_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("arcsin(2*x-1)", &mut direct_ctx).unwrap();
        let direct_result = unit_interval_bounded_inverse_trig_derivative_presentation(
            &mut direct_ctx,
            direct_target,
            "x",
        )
        .map(|compact| cas_ast::hold::wrap_hold(&mut direct_ctx, compact))
        .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
    }

    #[test]
    fn bounded_inverse_trig_route_preserves_projection_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("arccos(-x/sqrt(x^2+1))", &mut route_ctx).unwrap();
        let route_result =
            bounded_inverse_trig_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("arccos(-x/sqrt(x^2+1))", &mut direct_ctx).unwrap();
        let direct_result =
            bounded_inverse_trig_self_normalized_projection_derivative_presentation(
                &mut direct_ctx,
                direct_target,
                "x",
            )
            .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
    }
}

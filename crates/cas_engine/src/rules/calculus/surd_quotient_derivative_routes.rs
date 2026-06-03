//! Direct diff route for surd-quotient derivative presentation.
//!
//! This route keeps the existing `diff_rule` priority across arctangent
//! constant-radicand, self-normalized, and reciprocal self-normalized variants.
//! The presentation modules still own result construction and domain evidence.

use super::arctan_surd_derivative_presentation::{
    arctan_self_normalized_surd_reciprocal_compact_derivative,
    arctan_surd_quotient_compact_derivative, arctan_surd_quotient_scaled_compact_derivative,
    constant_scaled_arctan_surd_quotient_scaled_compact_derivative,
};
use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::inverse_surd_quotient_derivative_presentation::reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative;
use super::self_normalized_surd_quotient_derivative_presentation::direct_self_normalized_surd_quotient_post_calculus_presentation;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn constant_scaled_surd_quotient_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
        ctx, target, var_name,
    )
    .or_else(|| {
        constant_scaled_arctan_surd_quotient_scaled_compact_derivative(ctx, target, var_name)
    })
}

pub(super) fn constant_scaled_surd_quotient_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let result = constant_scaled_surd_quotient_derivative_route(ctx, target, &call.var_name)?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        Vec::new(),
    ))
}

pub(super) fn surd_quotient_derivative_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    arctan_surd_quotient_scaled_compact_derivative(ctx, target, var_name)
        .map(|result| (result, Vec::new()))
        .or_else(|| {
            arctan_surd_quotient_compact_derivative(ctx, target, var_name)
                .map(|result| (result, Vec::new()))
        })
        .or_else(|| {
            direct_self_normalized_surd_quotient_post_calculus_presentation(ctx, target, var_name)
                .map(|result| (result, Vec::new()))
        })
        .or_else(|| {
            let (result, required_condition) =
                arctan_self_normalized_surd_reciprocal_compact_derivative(ctx, target, var_name)?;
            Some((result, vec![required_condition]))
        })
}

pub(super) fn surd_quotient_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) =
        surd_quotient_derivative_route(ctx, target, &call.var_name)?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        required_conditions,
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::arctan_surd_derivative_presentation::{
        arctan_self_normalized_surd_reciprocal_compact_derivative,
        arctan_surd_quotient_compact_derivative, arctan_surd_quotient_scaled_compact_derivative,
        constant_scaled_arctan_surd_quotient_scaled_compact_derivative,
    };
    use super::super::inverse_surd_quotient_derivative_presentation::reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative;
    use super::super::self_normalized_surd_quotient_derivative_presentation::direct_self_normalized_surd_quotient_post_calculus_presentation;
    use super::{
        constant_scaled_surd_quotient_derivative_rewrite,
        constant_scaled_surd_quotient_derivative_route, surd_quotient_derivative_rewrite,
        surd_quotient_derivative_route,
    };
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn surd_quotient_route_preserves_scaled_arctan_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut route_ctx).unwrap();
        let (route_result, route_conditions) =
            surd_quotient_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut direct_ctx).unwrap();
        let direct_result =
            arctan_surd_quotient_scaled_compact_derivative(&mut direct_ctx, direct_target, "x")
                .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
        assert!(route_conditions.is_empty());
    }

    #[test]
    fn constant_scaled_surd_quotient_route_preserves_reciprocal_constant_scaled_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("1/2*arcsin(x/sqrt(4))", &mut route_ctx).unwrap();
        let route_result =
            constant_scaled_surd_quotient_derivative_route(&mut route_ctx, route_target, "x")
                .unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("1/2*arcsin(x/sqrt(4))", &mut direct_ctx).unwrap();
        let direct_result =
            reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
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
    fn constant_scaled_surd_quotient_route_preserves_scaled_arctan_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("7*arctan((2*x+1)/sqrt(3))/sqrt(3)", &mut route_ctx).unwrap();
        let route_result =
            constant_scaled_surd_quotient_derivative_route(&mut route_ctx, route_target, "x")
                .unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("7*arctan((2*x+1)/sqrt(3))/sqrt(3)", &mut direct_ctx).unwrap();
        let direct_result = constant_scaled_arctan_surd_quotient_scaled_compact_derivative(
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
    fn constant_scaled_surd_quotient_rewrite_preserves_result_and_empty_conditions() {
        let mut ctx = Context::new();
        let target = parse("1/2*arcsin(x/sqrt(4))", &mut ctx).unwrap();
        let route_result =
            constant_scaled_surd_quotient_derivative_route(&mut ctx, target, "x").unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite =
            constant_scaled_surd_quotient_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, route_result)
        );
        assert!(rewrite.required_conditions.is_empty());
    }

    #[test]
    fn surd_quotient_route_preserves_unscaled_arctan_hold() {
        let mut route_ctx = Context::new();
        let route_target = parse("arctan((2*x+2)/sqrt(6))", &mut route_ctx).unwrap();
        let (route_result, route_conditions) =
            surd_quotient_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("arctan((2*x+2)/sqrt(6))", &mut direct_ctx).unwrap();
        let direct_result =
            arctan_surd_quotient_compact_derivative(&mut direct_ctx, direct_target, "x").unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
        assert!(route_conditions.is_empty());
    }

    #[test]
    fn surd_quotient_route_preserves_self_normalized_result() {
        let mut route_ctx = Context::new();
        let route_target = parse("arctan(x/sqrt(x^2+1))", &mut route_ctx).unwrap();
        let (route_result, route_conditions) =
            surd_quotient_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("arctan(x/sqrt(x^2+1))", &mut direct_ctx).unwrap();
        let direct_result = direct_self_normalized_surd_quotient_post_calculus_presentation(
            &mut direct_ctx,
            direct_target,
            "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
        assert!(route_conditions.is_empty());
    }

    #[test]
    fn surd_quotient_route_preserves_reciprocal_condition() {
        let mut route_ctx = Context::new();
        let route_target = parse("arctan(sqrt(x^2+1)/x)", &mut route_ctx).unwrap();
        let (route_result, route_conditions) =
            surd_quotient_derivative_route(&mut route_ctx, route_target, "x").unwrap();

        let mut direct_ctx = Context::new();
        let direct_target = parse("arctan(sqrt(x^2+1)/x)", &mut direct_ctx).unwrap();
        let (direct_result, direct_condition) =
            arctan_self_normalized_surd_reciprocal_compact_derivative(
                &mut direct_ctx,
                direct_target,
                "x",
            )
            .unwrap();

        assert_eq!(
            rendered(&route_ctx, route_result),
            rendered(&direct_ctx, direct_result)
        );
        assert_eq!(route_conditions.len(), 1);
        assert_eq!(
            route_conditions[0].display(&route_ctx),
            direct_condition.display(&direct_ctx)
        );
    }

    #[test]
    fn surd_quotient_rewrite_preserves_reciprocal_condition() {
        let mut ctx = Context::new();
        let target = parse("arctan(sqrt(x^2+1)/x)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = surd_quotient_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "-1 / (sqrt(x^2 + 1) * (2 * x^2 + 1))"
        );
        assert_eq!(rewrite.required_conditions.len(), 1);
        assert_eq!(rewrite.required_conditions[0].display(&ctx), "x ≠ 0");
    }
}

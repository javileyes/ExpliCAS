//! Early integration routes for derivative-cofactor patterns.
//!
//! These routes own family-specific detection, presentation polish, and domain
//! conditions before the generic integration pipeline runs.

use super::arctan_additive_result_presentation::compact_arctan_additive_terms_for_calculus_presentation;
use super::arctan_polynomial_integrand_presentation::polynomial_times_arctan_affine_integrand_for_diff_shortcut;
use super::integration::integrate_rewrite_with_conditions;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn polynomial_trig_reciprocal_derivative_root_gate_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
) -> Option<Rewrite> {
    let (result, required_condition) =
        polynomial_trig_reciprocal_derivative_root_gate_route(ctx, call.target, &call.var_name)?;
    Some(integrate_rewrite_with_conditions(
        ctx,
        call,
        result,
        std::iter::once(required_condition),
    ))
}

pub(super) fn polynomial_trig_reciprocal_derivative_root_gate_route(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let (mut result, required_nonzero) =
        cas_math::symbolic_integration_support::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            ctx,
            target,
            var_name,
        )?;

    if polynomial_times_arctan_affine_integrand_for_diff_shortcut(ctx, target, var_name) {
        if let Some(compact) =
            compact_arctan_additive_terms_for_calculus_presentation(ctx, result, var_name)
        {
            result = compact;
        }
    }

    Some((result, crate::ImplicitCondition::NonZero(required_nonzero)))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn rendered_required_conditions(ctx: &Context, rewrite: &crate::rule::Rewrite) -> Vec<String> {
        rewrite
            .required_conditions
            .iter()
            .map(|condition| condition.display(ctx))
            .collect()
    }

    #[test]
    fn polynomial_trig_reciprocal_route_preserves_result_and_nonzero_condition() {
        let mut ctx = Context::new();
        let target = parse("(2*x+1)*sin(x^2+x)/cos(x^2+x)^2", &mut ctx).unwrap();
        let (result, required_condition) =
            super::polynomial_trig_reciprocal_derivative_root_gate_route(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(rendered(&ctx, result), "sec(x^2 + x)");
        let crate::ImplicitCondition::NonZero(required_nonzero) = required_condition else {
            panic!("expected NonZero condition");
        };
        assert_eq!(rendered(&ctx, required_nonzero), "cos(x^2 + x)");
    }

    #[test]
    fn polynomial_trig_reciprocal_route_rewrite_preserves_condition() {
        let mut ctx = Context::new();
        let target = parse("(2*x+1)*sin(x^2+x)/cos(x^2+x)^2", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite =
            super::polynomial_trig_reciprocal_derivative_root_gate_rewrite(&mut ctx, &call)
                .unwrap();

        assert_eq!(rendered(&ctx, rewrite.new_expr), "sec(x^2 + x)");
        assert_eq!(
            rendered_required_conditions(&ctx, &rewrite),
            vec!["cos(x^2 + x) ≠ 0"]
        );
    }
}

//! Compact inverse/surd derivative routes kept in `DiffRule` priority order.
//!
//! This boundary owns only the local route ordering and rewrite finalization.
//! The family-specific presentation and domain policies stay in their modules.

use cas_ast::{Context, ExprId};

use crate::symbolic_calculus_call_support::NamedVarCall;
use crate::Rewrite;

use super::asinh_surd_derivative_presentation::asinh_surd_quotient_compact_derivative;
use super::bounded_inverse_trig_derivative_routes::bounded_inverse_trig_derivative_route;
use super::diff_rule_support::finalize_diff_rewrite_with_conditions;

pub(super) fn compact_inverse_surd_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let result = bounded_inverse_trig_derivative_route(ctx, target, &call.var_name)
        .or_else(|| asinh_surd_quotient_compact_derivative(ctx, target, &call.var_name))?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        Vec::new(),
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use crate::symbolic_calculus_call_support::NamedVarCall;

    use super::super::asinh_surd_derivative_presentation::asinh_surd_quotient_compact_derivative;
    use super::super::bounded_inverse_trig_derivative_routes::bounded_inverse_trig_derivative_route;
    use super::compact_inverse_surd_derivative_rewrite;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_inverse_surd_rewrite_preserves_bounded_inverse_trig_result() {
        let mut ctx = Context::new();
        let target = parse("arcsin(x/sqrt(4))", &mut ctx).unwrap();
        let route_result = bounded_inverse_trig_derivative_route(&mut ctx, target, "x").unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = compact_inverse_surd_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, route_result)
        );
        assert!(rewrite.required_conditions.is_empty());
    }

    #[test]
    fn compact_inverse_surd_rewrite_preserves_asinh_surd_fallback_result() {
        let mut ctx = Context::new();
        let target = parse("asinh((x^2+x+1)/sqrt(7))", &mut ctx).unwrap();
        let route_result = asinh_surd_quotient_compact_derivative(&mut ctx, target, "x").unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite = compact_inverse_surd_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, route_result)
        );
        assert!(rewrite.required_conditions.is_empty());
    }
}

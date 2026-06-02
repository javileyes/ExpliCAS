//! Post-calculus presentation for polynomial arctan by-parts integrations.
//!
//! This keeps the source-side by-parts gate with the arctan result compaction it
//! enables. The integration rule still owns when the already-computed result is
//! preserved.

use super::arctan_additive_result_presentation::compact_arctan_additive_terms_for_calculus_presentation;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn compact_polynomial_arctan_by_parts_result_for_integration_presentation(
    ctx: &mut Context,
    var_name: &str,
    result: ExprId,
) -> Option<ExprId> {
    compact_arctan_additive_terms_for_calculus_presentation(ctx, result, var_name)
}

pub(super) fn try_compact_polynomial_arctan_by_parts_result_for_post_calculus_presentation(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
) -> Option<ExprId> {
    if cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
        ctx,
        call.target,
        &call.var_name,
    ) {
        return compact_polynomial_arctan_by_parts_result_for_integration_presentation(
            ctx,
            &call.var_name,
            result,
        );
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::symbolic_calculus_call_support::try_extract_integrate_call;
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::integration::integrate;
    use super::compact_polynomial_arctan_by_parts_result_for_integration_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn polynomial_arctan_by_parts_presentation_compacts_result_for_integrate_call() {
        let mut ctx = Context::new();
        let expr = parse("integrate(x^2*arctan(1-x), x)", &mut ctx).unwrap();
        let call = try_extract_integrate_call(&ctx, expr).unwrap();
        let raw_result = integrate(&mut ctx, call.target, &call.var_name).unwrap();
        let compact = compact_polynomial_arctan_by_parts_result_for_integration_presentation(
            &mut ctx,
            &call.var_name,
            raw_result,
        )
        .unwrap();

        assert_eq!(rendered(&ctx, compact).matches("arctan(1 - x)").count(), 1);
    }
}

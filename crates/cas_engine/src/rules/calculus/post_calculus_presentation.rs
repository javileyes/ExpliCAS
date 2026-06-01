use crate::symbolic_calculus_call_support::try_extract_integrate_call;
use cas_ast::{Constant, Context, Expr, ExprId};

use super::diff_post_calculus_presentation::try_diff_post_calculus_presentation;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::result_presentation::try_integrate_post_calculus_presentation;

pub(crate) fn try_post_calculus_presentation(
    ctx: &mut Context,
    source: ExprId,
    result: ExprId,
) -> Option<ExprId> {
    let unwrapped_result = unwrap_internal_hold_for_calculus(ctx, result);
    if matches!(
        ctx.get(unwrapped_result),
        Expr::Constant(Constant::Undefined)
    ) {
        return None;
    }

    if let Some(call) = try_extract_integrate_call(ctx, source) {
        if let Some(compact) = try_integrate_post_calculus_presentation(ctx, &call, result) {
            return Some(compact);
        }
    }

    try_diff_post_calculus_presentation(ctx, source, result)
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::try_post_calculus_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn post_calculus_presentation_compacts_nested_integral_source() {
        let mut ctx = Context::new();
        let source = parse(
            "diff(integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x), x)",
            &mut ctx,
        )
        .unwrap();
        let result = parse(
            "((3*x+1)^(1/2) * sin((3*x+1)^(1/2)) * 3)/(cos((3*x+1)^(1/2)) * (6*x+2))",
            &mut ctx,
        )
        .unwrap();
        let compact = try_post_calculus_presentation(&mut ctx, source, result).unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }
}

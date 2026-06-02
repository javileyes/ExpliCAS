use crate::symbolic_calculus_call_support::try_extract_integrate_call;
use cas_ast::{Constant, Context, Expr, ExprId};

use super::diff_post_calculus_presentation::try_diff_post_calculus_presentation;
use super::integrate_post_calculus_presentation::try_integrate_post_calculus_presentation;
use super::presentation_utils::unwrap_internal_hold_for_calculus;

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

use cas_ast::{Context, ExprId};

use super::diff_integral_source_post_calculus_presentation::try_diff_integral_source_post_calculus_presentation;
use super::sqrt_log_family_post_calculus_presentation::sqrt_log_family_post_calculus_presentation;

pub(super) fn result_sensitive_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        try_diff_integral_source_post_calculus_presentation(ctx, target, result, var_name)
    {
        return Some(compact);
    }

    sqrt_log_family_post_calculus_presentation(ctx, target, result, var_name)
}

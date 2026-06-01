use cas_ast::{Context, ExprId};

use super::arctan_surd_derivative_presentation::arctan_self_normalized_surd_quotient_compact_derivative;
use super::atanh_surd_derivative_presentation::atanh_self_normalized_surd_quotient_compact_derivative;

pub(super) fn direct_self_normalized_surd_quotient_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        arctan_self_normalized_surd_quotient_compact_derivative(ctx, target, var_name)
    {
        return Some(compact);
    }

    atanh_self_normalized_surd_quotient_compact_derivative(ctx, target, var_name)
}

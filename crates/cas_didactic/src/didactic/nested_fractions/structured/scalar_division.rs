use super::super::SubStep;
use cas_ast::{Context, ExprId};

pub(super) fn generate_sum_over_scalar_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let _ = (ctx, before_expr, after_expr, hints);
    Vec::new()
}

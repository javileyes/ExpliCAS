use super::SubStep;
use cas_ast::{Context, ExprId};

pub(super) fn get_standalone_substeps(
    ctx: &Context,
    original_expr: ExprId,
    standalone_fraction_sum_substeps: fn(&Context, ExprId) -> Vec<SubStep>,
) -> Vec<SubStep> {
    standalone_fraction_sum_substeps(ctx, original_expr)
}

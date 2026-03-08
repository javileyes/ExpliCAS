use super::super::fraction_sum_analysis::FractionSumInfo;
use cas_ast::{Context, ExprId};

pub(super) fn prepare_fraction_sum_context(
    ctx: &Context,
    original_expr: ExprId,
    collect_primary_fraction_sums: fn(&Context, ExprId) -> Vec<FractionSumInfo>,
) -> Vec<FractionSumInfo> {
    collect_primary_fraction_sums(ctx, original_expr)
}

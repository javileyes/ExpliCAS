use super::super::super::SubStep;
use super::FractionSumInfo;
use cas_ast::{Context, ExprId};

pub(super) fn standalone_fraction_sum_substeps(
    ctx: &Context,
    original_expr: ExprId,
    collect_primary_fraction_sums: fn(&Context, ExprId) -> Vec<FractionSumInfo>,
    extend_primary_fraction_sum_substeps: fn(&mut Vec<SubStep>, &[FractionSumInfo]),
) -> Vec<SubStep> {
    let unique_fraction_sums = collect_primary_fraction_sums(ctx, original_expr);
    let mut sub_steps = Vec::new();
    extend_primary_fraction_sum_substeps(&mut sub_steps, &unique_fraction_sums);
    sub_steps
}

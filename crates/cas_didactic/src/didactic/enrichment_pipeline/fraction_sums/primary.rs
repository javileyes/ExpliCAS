use super::super::super::fraction_steps::generate_fraction_sum_substeps;
use super::super::super::fraction_sum_analysis::{find_all_fraction_sums, FractionSumInfo};
use super::super::super::SubStep;
use super::{collect, extend};
use cas_ast::{Context, ExprId};

pub(super) fn extend_primary_fraction_sum_substeps(
    sub_steps: &mut Vec<SubStep>,
    unique_fraction_sums: &[FractionSumInfo],
) {
    extend::extend_primary_fraction_sum_substeps(
        sub_steps,
        unique_fraction_sums,
        generate_fraction_sum_substeps,
    )
}

pub(super) fn collect_primary_fraction_sums(
    ctx: &Context,
    original_expr: ExprId,
) -> Vec<FractionSumInfo> {
    collect::collect_primary_fraction_sums(ctx, original_expr, find_all_fraction_sums)
}

mod collect;
mod exponent;
mod extend;
mod primary;
mod standalone;

use super::super::fraction_sum_analysis::FractionSumInfo;
use super::super::SubStep;
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

pub(super) fn extend_primary_fraction_sum_substeps(
    sub_steps: &mut Vec<SubStep>,
    unique_fraction_sums: &[FractionSumInfo],
) {
    primary::extend_primary_fraction_sum_substeps(sub_steps, unique_fraction_sums)
}

pub(super) fn extend_exponent_fraction_sum_substeps(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
    unique_fraction_sums: &[FractionSumInfo],
    sub_steps: &mut Vec<SubStep>,
) {
    exponent::extend_exponent_fraction_sum_substeps(
        ctx,
        steps,
        step_idx,
        unique_fraction_sums,
        sub_steps,
    )
}

pub(super) fn standalone_fraction_sum_substeps(
    ctx: &Context,
    original_expr: ExprId,
) -> Vec<SubStep> {
    standalone::standalone_fraction_sum_substeps(
        ctx,
        original_expr,
        collect_primary_fraction_sums,
        extend_primary_fraction_sum_substeps,
    )
}

pub(super) fn collect_primary_fraction_sums(
    ctx: &Context,
    original_expr: ExprId,
) -> Vec<FractionSumInfo> {
    primary::collect_primary_fraction_sums(ctx, original_expr)
}

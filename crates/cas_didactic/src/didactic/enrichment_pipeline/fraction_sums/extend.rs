mod exponent;
mod primary;

use super::{FractionSumInfo, SubStep};
use crate::cas_solver::Step;
use cas_ast::Context;

pub(super) fn extend_primary_fraction_sum_substeps(
    sub_steps: &mut Vec<SubStep>,
    unique_fraction_sums: &[FractionSumInfo],
    generate_fraction_sum_substeps: fn(&FractionSumInfo) -> Vec<SubStep>,
) {
    primary::extend_primary_fraction_sum_substeps(
        sub_steps,
        unique_fraction_sums,
        generate_fraction_sum_substeps,
    )
}

pub(super) fn extend_exponent_fraction_sum_substeps(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
    unique_fraction_sums: &[FractionSumInfo],
    sub_steps: &mut Vec<SubStep>,
    detect_exponent_fraction_change: fn(&Context, &[Step], usize) -> Option<FractionSumInfo>,
    generate_fraction_sum_substeps: fn(&FractionSumInfo) -> Vec<SubStep>,
) {
    exponent::extend_exponent_fraction_sum_substeps(
        ctx,
        steps,
        step_idx,
        unique_fraction_sums,
        sub_steps,
        detect_exponent_fraction_change,
        generate_fraction_sum_substeps,
    )
}

use super::super::super::fraction_steps::generate_fraction_sum_substeps;
use super::super::super::fraction_sum_analysis::{
    detect_exponent_fraction_change, FractionSumInfo,
};
use super::super::super::SubStep;
use super::extend;
use crate::cas_solver::Step;
use cas_ast::Context;

pub(super) fn extend_exponent_fraction_sum_substeps(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
    unique_fraction_sums: &[FractionSumInfo],
    sub_steps: &mut Vec<SubStep>,
) {
    extend::extend_exponent_fraction_sum_substeps(
        ctx,
        steps,
        step_idx,
        unique_fraction_sums,
        sub_steps,
        detect_exponent_fraction_change,
        generate_fraction_sum_substeps,
    )
}

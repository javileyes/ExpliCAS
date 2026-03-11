use super::{FractionSumInfo, SubStep};
use crate::runtime::Step;
use cas_ast::Context;

pub(super) fn extend_exponent_fraction_sum_substeps(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
    unique_fraction_sums: &[FractionSumInfo],
    sub_steps: &mut Vec<SubStep>,
    detect_exponent_fraction_change: fn(&Context, &[Step], usize) -> Option<FractionSumInfo>,
    generate_fraction_sum_substeps: fn(&FractionSumInfo) -> Vec<SubStep>,
) {
    if let Some(fraction_info) = detect_exponent_fraction_change(ctx, steps, step_idx) {
        if !unique_fraction_sums
            .iter()
            .any(|known| known.fractions == fraction_info.fractions)
        {
            sub_steps.extend(generate_fraction_sum_substeps(&fraction_info));
        }
    }
}

use super::{FractionSumInfo, SubStep};

pub(super) fn extend_primary_fraction_sum_substeps(
    sub_steps: &mut Vec<SubStep>,
    unique_fraction_sums: &[FractionSumInfo],
    generate_fraction_sum_substeps: fn(&FractionSumInfo) -> Vec<SubStep>,
) {
    for info in unique_fraction_sums {
        sub_steps.extend(generate_fraction_sum_substeps(info));
    }
}

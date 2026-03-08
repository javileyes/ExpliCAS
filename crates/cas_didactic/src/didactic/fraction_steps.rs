mod common_denominator;
mod render;

use super::fraction_sum_analysis::FractionSumInfo;
use super::SubStep;

/// Generate sub-steps explaining how fractions were summed
pub(crate) fn generate_fraction_sum_substeps(info: &FractionSumInfo) -> Vec<SubStep> {
    if info.fractions.len() < 2 {
        return Vec::new();
    }

    let common = common_denominator::build_common_denominator_data(
        info,
        super::format_fraction,
        super::lcm_bigint,
    );
    render::render_fraction_sum_substeps(info, common)
}

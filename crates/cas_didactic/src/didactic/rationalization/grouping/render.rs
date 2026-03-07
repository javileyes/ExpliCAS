mod conjugate;
mod difference;
mod group_terms;

use super::conjugate::GroupedRationalizationData;
use crate::didactic::SubStep;

pub(super) fn build_group_terms_substep(
    numerator_latex: &str,
    denominator_latex: &str,
    grouping: &GroupedRationalizationData,
) -> SubStep {
    group_terms::build_group_terms_substep(numerator_latex, denominator_latex, grouping)
}

pub(super) fn build_grouped_conjugate_substep(grouping: &GroupedRationalizationData) -> SubStep {
    conjugate::build_grouped_conjugate_substep(grouping)
}

pub(super) fn build_difference_of_squares_substep(
    grouping: &GroupedRationalizationData,
    after_denominator_latex: &str,
) -> SubStep {
    difference::build_difference_of_squares_substep(grouping, after_denominator_latex)
}

use super::super::conjugate::GroupedRationalizationData;
use crate::didactic::SubStep;

pub(super) fn build_grouped_conjugate_substep(grouping: &GroupedRationalizationData) -> SubStep {
    SubStep {
        description: "Multiplicar por el conjugado".to_string(),
        before_expr: grouping.grouped_sum.clone(),
        after_expr: grouping.grouped_conjugate.clone(),
        before_latex: None,
        after_latex: None,
    }
}

use super::super::conjugate::GroupedRationalizationData;
use crate::didactic::SubStep;

pub(super) fn build_group_terms_substep(
    numerator_latex: &str,
    denominator_latex: &str,
    grouping: &GroupedRationalizationData,
) -> SubStep {
    SubStep {
        description: "Agrupar términos del denominador".to_string(),
        before_expr: format!("\\frac{{{}}}{{{}}}", numerator_latex, denominator_latex),
        after_expr: format!("\\frac{{{}}}{{{}}}", numerator_latex, grouping.grouped_sum),
        before_latex: None,
        after_latex: None,
    }
}

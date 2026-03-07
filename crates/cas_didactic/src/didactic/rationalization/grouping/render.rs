use super::conjugate::GroupedRationalizationData;
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

pub(super) fn build_grouped_conjugate_substep(grouping: &GroupedRationalizationData) -> SubStep {
    SubStep {
        description: "Multiplicar por el conjugado".to_string(),
        before_expr: grouping.grouped_sum.clone(),
        after_expr: grouping.grouped_conjugate.clone(),
        before_latex: None,
        after_latex: None,
    }
}

pub(super) fn build_difference_of_squares_substep(
    grouping: &GroupedRationalizationData,
    after_denominator_latex: &str,
) -> SubStep {
    SubStep {
        description: "Diferencia de cuadrados".to_string(),
        before_expr: format!(
            "{}^2 - ({})^2",
            grouping.grouped_sum_with_parens,
            extract_last_term(grouping)
        ),
        after_expr: after_denominator_latex.to_string(),
        before_latex: None,
        after_latex: None,
    }
}

fn extract_last_term(grouping: &GroupedRationalizationData) -> &str {
    grouping
        .grouped_conjugate
        .rsplit(" - ")
        .next()
        .unwrap_or(grouping.grouped_conjugate.as_str())
}

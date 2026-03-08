use crate::didactic::SubStep;

pub(super) fn build_binomial_conjugate_substep(
    numerator_latex: &str,
    denominator_latex: &str,
    conjugate: &str,
) -> SubStep {
    SubStep {
        description: "Denominador binomial con radical".to_string(),
        before_expr: format!("\\frac{{{}}}{{{}}}", numerator_latex, denominator_latex),
        after_expr: format!("\\text{{Conjugado: }} {}", conjugate),
        before_latex: None,
        after_latex: None,
    }
}

pub(super) fn build_binomial_product_substep(
    numerator_latex: &str,
    denominator_latex: &str,
    conjugate: &str,
    after_num_latex: &str,
    after_den_latex: &str,
) -> SubStep {
    SubStep {
        description: "(a+b)(a-b) = a² - b²".to_string(),
        before_expr: format!(
            "\\frac{{({}) \\cdot ({})}}{{{}  \\cdot ({})}}",
            numerator_latex, conjugate, denominator_latex, conjugate
        ),
        after_expr: format!("\\frac{{{}}}{{{}}}", after_num_latex, after_den_latex),
        before_latex: None,
        after_latex: None,
    }
}

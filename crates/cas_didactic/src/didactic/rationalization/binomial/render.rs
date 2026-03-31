use crate::didactic::SubStep;

fn humanize_even_literal_square(input: &str) -> String {
    input
        .replace("{(-1)}^{2}", "1^{2}")
        .replace("((-1))^2", "1^2")
        .replace("(-1)^2", "1^2")
}

pub(super) fn build_binomial_conjugate_substep(
    denominator_latex: &str,
    conjugate: &str,
) -> SubStep {
    SubStep::new(
        "Cambiar el signo para formar el conjugado",
        denominator_latex,
        conjugate,
    )
    .with_before_latex(denominator_latex)
    .with_after_latex(conjugate)
}

pub(super) fn build_binomial_multiply_both_sides_substep(
    numerator_latex: &str,
    denominator_latex: &str,
    conjugate: &str,
) -> SubStep {
    SubStep::new(
        "Multiplicar numerador y denominador por ese conjugado",
        format!("({numerator_latex})/({denominator_latex})"),
        format!(
            "(({numerator_latex}) · ({conjugate}))/(({denominator_latex}) · ({conjugate}))"
        ),
    )
    .with_before_latex(format!("\\frac{{{numerator_latex}}}{{{denominator_latex}}}"))
    .with_after_latex(format!(
        "\\frac{{({numerator_latex}) \\cdot ({conjugate})}}{{({denominator_latex}) \\cdot ({conjugate})}}"
    ))
}

pub(super) fn build_binomial_product_substep(
    numerator_latex: &str,
    denominator_latex: &str,
    conjugate: &str,
    after_num_latex: &str,
    after_den_latex: &str,
) -> SubStep {
    let after_den_latex = humanize_even_literal_square(after_den_latex);
    SubStep::new(
        "En el denominador aparece una diferencia de cuadrados",
        format!(
            "(({numerator_latex}) · ({conjugate}))/(({denominator_latex}) · ({conjugate}))"
        ),
        format!("({after_num_latex})/({after_den_latex})"),
    )
    .with_before_latex(format!(
        "\\frac{{({numerator_latex}) \\cdot ({conjugate})}}{{({denominator_latex}) \\cdot ({conjugate})}}"
    ))
    .with_after_latex(format!("\\frac{{{after_num_latex}}}{{{after_den_latex}}}"))
}

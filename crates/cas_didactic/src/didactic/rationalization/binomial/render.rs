use crate::didactic::SubStep;

fn humanize_even_literal_square(input: &str) -> String {
    input
        .replace("{(-1)}^{2}", "1^{2}")
        .replace("((-1))^2", "1^2")
        .replace("(-1)^2", "1^2")
}

fn multiply_factor_text(left: &str, right: &str) -> String {
    match (left.trim(), right.trim()) {
        ("1", other) => format!("({other})"),
        (other, "1") => format!("({other})"),
        (lhs, rhs) => format!("({lhs}) · ({rhs})"),
    }
}

fn fraction_text(numerator: &str, denominator: &str) -> String {
    format!("({numerator})/({denominator})")
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
    let before_fraction = fraction_text(numerator_latex, denominator_latex);
    let after_numerator = multiply_factor_text(numerator_latex, conjugate);
    let after_denominator = multiply_factor_text(denominator_latex, conjugate);
    SubStep::new(
        "Multiplicar numerador y denominador por ese conjugado",
        before_fraction,
        fraction_text(&after_numerator, &after_denominator),
    )
    .with_before_latex(format!(
        "\\frac{{{numerator_latex}}}{{{denominator_latex}}}"
    ))
    .with_after_latex(format!(
        "\\frac{{{}}}{{{}}}",
        after_numerator.replace('·', "\\cdot"),
        after_denominator.replace('·', "\\cdot")
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
    let before_numerator = multiply_factor_text(numerator_latex, conjugate);
    let before_denominator = multiply_factor_text(denominator_latex, conjugate);
    SubStep::new(
        "En el denominador aparece una diferencia de cuadrados",
        fraction_text(&before_numerator, &before_denominator),
        format!("({after_num_latex})/({after_den_latex})"),
    )
    .with_before_latex(format!(
        "\\frac{{{}}}{{{}}}",
        before_numerator.replace('·', "\\cdot"),
        before_denominator.replace('·', "\\cdot")
    ))
    .with_after_latex(format!("\\frac{{{after_num_latex}}}{{{after_den_latex}}}"))
}

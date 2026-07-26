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

/// Each hole receives its OWN surface: the plain strings go to the plain
/// fields and the LaTeX strings to the latex fields. The audit found this
/// family publishing `\sqrt{y} - 1` in the plain hole — the constructor took
/// one string per SIDE and reused it for both surfaces, so nothing stopped
/// LaTeX from reaching the CLI reader. Two strings per side make the crossing
/// visible at the call site; the structural fix (a `Rendered { plain, latex }`
/// pair type, C1.8 design §2) stays noted for the class.
pub(super) fn build_binomial_conjugate_substep(
    denominator_plain: &str,
    denominator_latex: &str,
    conjugate_plain: &str,
    conjugate_latex: &str,
) -> SubStep {
    SubStep::keyed(
        "rationalize.form_conjugate",
        vec![],
        denominator_plain,
        conjugate_plain,
    )
    .with_before_latex(denominator_latex)
    .with_after_latex(conjugate_latex)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_binomial_multiply_both_sides_substep(
    numerator_plain: &str,
    numerator_latex: &str,
    denominator_plain: &str,
    denominator_latex: &str,
    conjugate_plain: &str,
    conjugate_latex: &str,
) -> SubStep {
    let after_numerator_plain = multiply_factor_text(numerator_plain, conjugate_plain);
    let after_denominator_plain = multiply_factor_text(denominator_plain, conjugate_plain);
    let after_numerator = multiply_factor_text(numerator_latex, conjugate_latex);
    let after_denominator = multiply_factor_text(denominator_latex, conjugate_latex);
    SubStep::keyed(
        "rationalize.multiply_by_conjugate_both",
        vec![],
        fraction_text(numerator_plain, denominator_plain),
        fraction_text(&after_numerator_plain, &after_denominator_plain),
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

#[allow(clippy::too_many_arguments)]
pub(super) fn build_binomial_product_substep(
    numerator_plain: &str,
    numerator_latex: &str,
    denominator_plain: &str,
    denominator_latex: &str,
    conjugate_plain: &str,
    conjugate_latex: &str,
    after_num_plain: &str,
    after_num_latex: &str,
    after_den_plain: &str,
    after_den_latex: &str,
) -> SubStep {
    let after_den_plain = humanize_even_literal_square(after_den_plain);
    let after_den_latex = humanize_even_literal_square(after_den_latex);
    let before_numerator_plain = multiply_factor_text(numerator_plain, conjugate_plain);
    let before_denominator_plain = multiply_factor_text(denominator_plain, conjugate_plain);
    let before_numerator = multiply_factor_text(numerator_latex, conjugate_latex);
    let before_denominator = multiply_factor_text(denominator_latex, conjugate_latex);
    SubStep::keyed(
        "rationalize.denominator_difference_of_squares",
        vec![],
        fraction_text(&before_numerator_plain, &before_denominator_plain),
        fraction_text(after_num_plain, &after_den_plain),
    )
    .with_before_latex(format!(
        "\\frac{{{}}}{{{}}}",
        before_numerator.replace('·', "\\cdot"),
        before_denominator.replace('·', "\\cdot")
    ))
    .with_after_latex(format!("\\frac{{{after_num_latex}}}{{{after_den_latex}}}"))
}

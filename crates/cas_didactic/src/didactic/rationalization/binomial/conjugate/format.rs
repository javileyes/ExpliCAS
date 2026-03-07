use super::extract::BinomialTerms;

pub(super) fn render_binomial_conjugate(terms: &BinomialTerms, denominator_latex: &str) -> String {
    if terms.term_b.is_empty() {
        denominator_latex.to_string()
    } else if terms.is_original_minus {
        format!("{} + {}", terms.term_a, terms.term_b)
    } else {
        format!("{} - {}", terms.term_a, terms.term_b)
    }
}

pub(super) fn format_negative_number_latex(number: &num_rational::BigRational) -> String {
    let abs_number = -number;
    if abs_number.is_integer() {
        format!("{}", abs_number.numer())
    } else {
        format!("\\frac{{{}}}{{{}}}", abs_number.numer(), abs_number.denom())
    }
}

use super::BinomialTerms;

pub(super) fn extract_fallback_binomial_terms(denominator_latex: &str) -> BinomialTerms {
    BinomialTerms {
        term_a: denominator_latex.to_string(),
        term_b: String::new(),
        is_original_minus: false,
    }
}

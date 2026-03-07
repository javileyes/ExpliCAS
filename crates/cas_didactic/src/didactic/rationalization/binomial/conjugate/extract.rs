mod add_case;
mod fallback;
mod sub_case;

use cas_ast::{Context, Expr, ExprId};

pub(super) struct BinomialTerms {
    pub(super) term_a: String,
    pub(super) term_b: String,
    pub(super) is_original_minus: bool,
}

pub(super) fn extract_binomial_terms(
    context: &Context,
    denominator: ExprId,
    hints: &cas_formatter::DisplayContext,
    denominator_latex: &str,
    rationalization_latex: fn(&Context, &cas_formatter::DisplayContext, ExprId) -> String,
    format_negative_number_latex: fn(&num_rational::BigRational) -> String,
) -> BinomialTerms {
    match context.get(denominator) {
        Expr::Add(left, right) => add_case::extract_add_binomial_terms(
            context,
            hints,
            *left,
            *right,
            rationalization_latex,
            format_negative_number_latex,
        ),
        Expr::Sub(left, right) => sub_case::extract_sub_binomial_terms(
            context,
            hints,
            *left,
            *right,
            rationalization_latex,
        ),
        _ => fallback::extract_fallback_binomial_terms(denominator_latex),
    }
}

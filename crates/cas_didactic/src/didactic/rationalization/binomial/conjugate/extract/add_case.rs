use super::BinomialTerms;
use cas_ast::{Context, Expr, ExprId};

pub(super) fn extract_add_binomial_terms(
    context: &Context,
    hints: &cas_formatter::DisplayContext,
    left: ExprId,
    right: ExprId,
    rationalization_latex: fn(&Context, &cas_formatter::DisplayContext, ExprId) -> String,
    format_negative_number_latex: fn(&num_rational::BigRational) -> String,
) -> BinomialTerms {
    match context.get(right) {
        Expr::Neg(inner) => BinomialTerms {
            term_a: rationalization_latex(context, hints, left),
            term_b: rationalization_latex(context, hints, *inner),
            is_original_minus: true,
        },
        Expr::Number(number) if num_traits::Signed::is_negative(number) => BinomialTerms {
            term_a: rationalization_latex(context, hints, left),
            term_b: format_negative_number_latex(number),
            is_original_minus: true,
        },
        _ => BinomialTerms {
            term_a: rationalization_latex(context, hints, left),
            term_b: rationalization_latex(context, hints, right),
            is_original_minus: false,
        },
    }
}

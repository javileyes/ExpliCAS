use super::BinomialTerms;
use cas_ast::{Context, ExprId};

pub(super) fn extract_sub_binomial_terms(
    context: &Context,
    hints: &cas_formatter::DisplayContext,
    left: ExprId,
    right: ExprId,
    rationalization_latex: fn(&Context, &cas_formatter::DisplayContext, ExprId) -> String,
) -> BinomialTerms {
    BinomialTerms {
        term_a: rationalization_latex(context, hints, left),
        term_b: rationalization_latex(context, hints, right),
        is_original_minus: true,
    }
}

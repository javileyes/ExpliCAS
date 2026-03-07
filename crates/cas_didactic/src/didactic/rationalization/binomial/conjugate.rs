mod extract;
mod format;

use cas_ast::{Context, ExprId};

pub(super) fn build_binomial_conjugate(
    context: &Context,
    denominator: ExprId,
    denominator_latex: &str,
    hints: &cas_formatter::DisplayContext,
) -> String {
    let terms = extract::extract_binomial_terms(
        context,
        denominator,
        hints,
        denominator_latex,
        super::super::rationalization_latex,
        format::format_negative_number_latex,
    );
    format::render_binomial_conjugate(&terms, denominator_latex)
}

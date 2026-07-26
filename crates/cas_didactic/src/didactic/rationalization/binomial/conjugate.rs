mod extract;
mod format;

use cas_ast::{Context, ExprId};

/// Plain twin of [`build_binomial_conjugate`]: the SAME extraction walk with
/// the plain renderer, so the two surfaces cannot drift — the machinery is
/// renderer-parametric and runs once per surface.
pub(super) fn build_binomial_conjugate_plain(
    context: &Context,
    denominator: ExprId,
    denominator_plain: &str,
    hints: &cas_formatter::DisplayContext,
) -> String {
    let terms = extract::extract_binomial_terms(
        context,
        denominator,
        hints,
        denominator_plain,
        super::super::rationalization_display,
        format::format_negative_number_plain,
    );
    format::render_binomial_conjugate(&terms, denominator_plain)
}

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

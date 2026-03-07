use cas_ast::{Context, Expr, ExprId};
use num_traits::Signed;

pub(super) fn build_binomial_conjugate(
    context: &Context,
    denominator: ExprId,
    denominator_latex: &str,
    hints: &cas_formatter::DisplayContext,
) -> String {
    let (term_a, term_b, is_original_minus) =
        extract_binomial_terms(context, denominator, hints, denominator_latex);

    if term_b.is_empty() {
        denominator_latex.to_string()
    } else if is_original_minus {
        format!("{} + {}", term_a, term_b)
    } else {
        format!("{} - {}", term_a, term_b)
    }
}

fn extract_binomial_terms(
    context: &Context,
    denominator: ExprId,
    hints: &cas_formatter::DisplayContext,
    denominator_latex: &str,
) -> (String, String, bool) {
    match context.get(denominator) {
        Expr::Add(left, right) => match context.get(*right) {
            Expr::Neg(inner) => (
                super::super::rationalization_latex(context, hints, *left),
                super::super::rationalization_latex(context, hints, *inner),
                true,
            ),
            Expr::Number(number) if number.is_negative() => (
                super::super::rationalization_latex(context, hints, *left),
                format_negative_number_latex(number),
                true,
            ),
            _ => (
                super::super::rationalization_latex(context, hints, *left),
                super::super::rationalization_latex(context, hints, *right),
                false,
            ),
        },
        Expr::Sub(left, right) => (
            super::super::rationalization_latex(context, hints, *left),
            super::super::rationalization_latex(context, hints, *right),
            true,
        ),
        _ => (denominator_latex.to_string(), String::new(), false),
    }
}

fn format_negative_number_latex(number: &num_rational::BigRational) -> String {
    let abs_number = -number;
    if abs_number.is_integer() {
        format!("{}", abs_number.numer())
    } else {
        format!("\\frac{{{}}}{{{}}}", abs_number.numer(), abs_number.denom())
    }
}

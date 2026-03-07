use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, LaTeXExprWithHints};
use num_rational::BigRational;

pub(super) fn denesting_latex(ctx: &Context, id: ExprId) -> String {
    let hints = DisplayContext::with_root_index(2);
    LaTeXExprWithHints {
        context: ctx,
        id,
        hints: &hints,
    }
    .to_latex()
}

pub(super) fn format_rational_latex(value: &BigRational) -> String {
    if value.is_integer() {
        format!("{}", value.to_integer())
    } else {
        format!("\\frac{{{}}}{{{}}}", value.numer(), value.denom())
    }
}

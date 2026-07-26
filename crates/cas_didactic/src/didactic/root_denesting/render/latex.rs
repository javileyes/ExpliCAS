use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, DisplayExprWithHints, LaTeXExprWithHints};
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

/// Plain-text twin of [`denesting_latex`]. The substep's `before_expr`/`after_expr`
/// are the TEXT channel (CLI, and the `before`/`after` wire fields); LaTeX belongs
/// in `before_latex`/`after_latex`. Emitting LaTeX into the text fields used to
/// work only because the renderer let a LaTeX-ish fallback through untouched.
pub(super) fn denesting_display(ctx: &Context, id: ExprId) -> String {
    let hints = DisplayContext::with_root_index(2);
    format!(
        "{}",
        DisplayExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
    )
}

pub(super) fn format_rational_display(value: &BigRational) -> String {
    if value.is_integer() {
        format!("{}", value.to_integer())
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

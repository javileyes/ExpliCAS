use cas_ast::{Context, ExprId};

pub(super) fn rationalization_latex(
    ctx: &Context,
    hints: &cas_formatter::DisplayContext,
    id: ExprId,
) -> String {
    cas_formatter::LaTeXExprWithHints {
        context: ctx,
        id,
        hints,
    }
    .to_latex()
}

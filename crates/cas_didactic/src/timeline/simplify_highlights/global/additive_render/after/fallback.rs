use cas_ast::Context;
use cas_formatter::{HighlightColor, LaTeXExpr};

pub(super) fn render_after_additive_focus_fallback(
    context: &Context,
    global_after_expr: cas_ast::ExprId,
    _focus_after: cas_ast::ExprId,
    _highlight_color: HighlightColor,
) -> String {
    LaTeXExpr {
        context,
        id: global_after_expr,
    }
    .to_latex()
}

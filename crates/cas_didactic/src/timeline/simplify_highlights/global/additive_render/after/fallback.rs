use cas_ast::Context;
use cas_formatter::{HighlightColor, HighlightConfig, LaTeXExprHighlighted};

pub(super) fn render_after_additive_focus_fallback(
    context: &Context,
    global_after_expr: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    highlight_color: HighlightColor,
) -> String {
    let mut after_config = HighlightConfig::new();
    after_config.add(focus_after, highlight_color);
    LaTeXExprHighlighted {
        context,
        id: global_after_expr,
        highlights: &after_config,
    }
    .to_latex()
}

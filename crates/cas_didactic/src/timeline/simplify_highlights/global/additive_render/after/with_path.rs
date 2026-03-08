use cas_ast::{Context, ExprPath};
use cas_formatter::{DisplayContext, HighlightColor, StylePreferences};

pub(super) fn render_after_additive_focus_with_path(
    context: &Context,
    global_after_expr: cas_ast::ExprId,
    after_path: ExprPath,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
    highlight_color: HighlightColor,
    render_with_single_path: fn(
        &Context,
        cas_ast::ExprId,
        ExprPath,
        HighlightColor,
        &DisplayContext,
        &StylePreferences,
    ) -> String,
) -> String {
    render_with_single_path(
        context,
        global_after_expr,
        after_path,
        highlight_color,
        display_hints,
        style_prefs,
    )
}

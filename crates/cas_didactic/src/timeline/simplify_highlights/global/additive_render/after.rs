mod fallback;
mod with_path;

use cas_ast::Context;
use cas_formatter::{DisplayContext, HighlightColor, StylePreferences};

pub(super) fn render_after_additive_focus(
    context: &Context,
    global_after_expr: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
    render_with_single_path: fn(
        &Context,
        cas_ast::ExprId,
        cas_ast::ExprPath,
        HighlightColor,
        &DisplayContext,
        &StylePreferences,
    ) -> String,
) -> String {
    if let Some(after_path) =
        cas_formatter::path::diff_find_path_to_expr(context, global_after_expr, focus_after)
    {
        return with_path::render_after_additive_focus_with_path(
            context,
            global_after_expr,
            after_path,
            display_hints,
            style_prefs,
            HighlightColor::Green,
            render_with_single_path,
        );
    }

    fallback::render_after_additive_focus_fallback(
        context,
        global_after_expr,
        focus_after,
        HighlightColor::Green,
    )
}

use crate::runtime::{pathsteps_to_expr_path, Step};
use cas_ast::{Context, ExprPath};
use cas_formatter::{DisplayContext, HighlightColor, StylePreferences};

pub(super) fn render_before_additive_focus_fallback(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    step: &Step,
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
        global_before_expr,
        pathsteps_to_expr_path(step.path()),
        highlight_color,
        display_hints,
        style_prefs,
    )
}

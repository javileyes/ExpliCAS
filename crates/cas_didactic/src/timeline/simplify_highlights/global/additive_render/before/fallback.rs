use crate::runtime::{pathsteps_to_expr_path, Step};
use cas_ast::{Context, ExprPath};
use cas_formatter::path::{diff_find_path_to_expr, diff_find_paths_by_structure};
use cas_formatter::{DisplayContext, HighlightColor, StylePreferences};

#[allow(clippy::too_many_arguments)]
pub(super) fn render_before_additive_focus_fallback(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    focus_before: cas_ast::ExprId,
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
    let fallback_path = diff_find_path_to_expr(context, global_before_expr, focus_before)
        .or_else(|| {
            diff_find_paths_by_structure(context, global_before_expr, focus_before)
                .into_iter()
                .next()
        })
        .unwrap_or_else(|| pathsteps_to_expr_path(step.path()));

    render_with_single_path(
        context,
        global_before_expr,
        fallback_path,
        highlight_color,
        display_hints,
        style_prefs,
    )
}

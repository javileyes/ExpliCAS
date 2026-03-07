use cas_ast::{Context, ExprPath};
use cas_formatter::{DisplayContext, PathHighlightConfig, StylePreferences};

pub(super) fn render_before_additive_focus_with_paths(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    found_paths: &[ExprPath],
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
    build_before_additive_focus_config: fn(&[ExprPath]) -> PathHighlightConfig,
    render_with_paths: fn(
        &Context,
        cas_ast::ExprId,
        &PathHighlightConfig,
        &DisplayContext,
        &StylePreferences,
    ) -> String,
) -> String {
    let before_config = build_before_additive_focus_config(found_paths);
    render_with_paths(
        context,
        global_before_expr,
        &before_config,
        display_hints,
        style_prefs,
    )
}

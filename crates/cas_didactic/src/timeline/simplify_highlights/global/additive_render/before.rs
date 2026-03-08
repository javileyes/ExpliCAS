mod config;
mod fallback;
mod with_paths;

use cas_ast::{Context, ExprPath};
use cas_formatter::{DisplayContext, HighlightColor, PathHighlightConfig, StylePreferences};
use cas_solver::Step;

#[allow(clippy::too_many_arguments)]
pub(super) fn render_before_additive_focus(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    found_paths: &[ExprPath],
    step: &Step,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
    render_with_paths: fn(
        &Context,
        cas_ast::ExprId,
        &PathHighlightConfig,
        &DisplayContext,
        &StylePreferences,
    ) -> String,
    render_with_single_path: fn(
        &Context,
        cas_ast::ExprId,
        ExprPath,
        HighlightColor,
        &DisplayContext,
        &StylePreferences,
    ) -> String,
) -> String {
    if !found_paths.is_empty() {
        return with_paths::render_before_additive_focus_with_paths(
            context,
            global_before_expr,
            found_paths,
            display_hints,
            style_prefs,
            config::build_before_additive_focus_config,
            render_with_paths,
        );
    }

    fallback::render_before_additive_focus_fallback(
        context,
        global_before_expr,
        step,
        display_hints,
        style_prefs,
        HighlightColor::Red,
        render_with_single_path,
    )
}

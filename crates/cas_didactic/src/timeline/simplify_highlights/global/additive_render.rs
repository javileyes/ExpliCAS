use super::super::renderers::{render_with_paths, render_with_single_path};
use cas_ast::{Context, ExprPath};
use cas_formatter::{
    DisplayContext, HighlightColor, HighlightConfig, LaTeXExprHighlighted, PathHighlightConfig,
    StylePreferences,
};
use cas_solver::{pathsteps_to_expr_path, Step};

pub(super) fn render_before_additive_focus(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    found_paths: &[ExprPath],
    step: &Step,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    if !found_paths.is_empty() {
        let mut before_config = PathHighlightConfig::new();
        for path in found_paths.iter().cloned() {
            before_config.add(path, HighlightColor::Red);
        }
        return render_with_paths(
            context,
            global_before_expr,
            &before_config,
            display_hints,
            style_prefs,
        );
    }

    render_with_single_path(
        context,
        global_before_expr,
        pathsteps_to_expr_path(step.path()),
        HighlightColor::Red,
        display_hints,
        style_prefs,
    )
}

pub(super) fn render_after_additive_focus(
    context: &Context,
    global_after_expr: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    if let Some(after_path) =
        cas_formatter::path::diff_find_path_to_expr(context, global_after_expr, focus_after)
    {
        return render_with_single_path(
            context,
            global_after_expr,
            after_path,
            HighlightColor::Green,
            display_hints,
            style_prefs,
        );
    }

    let mut after_config = HighlightConfig::new();
    after_config.add(focus_after, HighlightColor::Green);
    LaTeXExprHighlighted {
        context,
        id: global_after_expr,
        highlights: &after_config,
    }
    .to_latex()
}

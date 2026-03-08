use cas_ast::{Context, ExprId, ExprPath};
use cas_formatter::{
    DisplayContext, HighlightColor, PathHighlightConfig, PathHighlightedLatexRenderer,
    StylePreferences,
};

pub(super) fn render_with_single_path(
    context: &Context,
    id: ExprId,
    path: ExprPath,
    color: HighlightColor,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    let mut config = PathHighlightConfig::new();
    config.add(path, color);
    render_with_paths(context, id, &config, display_hints, style_prefs)
}

pub(super) fn render_with_paths(
    context: &Context,
    id: ExprId,
    path_highlights: &PathHighlightConfig,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    PathHighlightedLatexRenderer {
        context,
        id,
        path_highlights,
        hints: Some(display_hints),
        style_prefs: Some(style_prefs),
    }
    .to_latex()
}

mod local_change;
mod paths;

use crate::cas_solver::Step;
use cas_ast::{Context, ExprId, ExprPath};
use cas_formatter::{DisplayContext, HighlightColor, PathHighlightConfig, StylePreferences};

pub(super) fn render_local_change_latex(
    context: &Context,
    step: &Step,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    local_change::render_local_change_latex(context, step, display_hints, style_prefs)
}

pub(super) fn render_with_single_path(
    context: &Context,
    id: ExprId,
    path: ExprPath,
    color: HighlightColor,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    paths::render_with_single_path(context, id, path, color, display_hints, style_prefs)
}

pub(super) fn render_with_paths(
    context: &Context,
    id: ExprId,
    path_highlights: &PathHighlightConfig,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    paths::render_with_paths(context, id, path_highlights, display_hints, style_prefs)
}

mod after;
mod before;

use super::super::renderers::{render_with_paths, render_with_single_path};
use crate::runtime::Step;
use cas_ast::{Context, ExprId, ExprPath};
use cas_formatter::{DisplayContext, StylePreferences};

pub(super) fn render_before_additive_focus(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    focus_before: ExprId,
    found_paths: &[ExprPath],
    step: &Step,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    before::render_before_additive_focus(
        context,
        global_before_expr,
        focus_before,
        found_paths,
        step,
        display_hints,
        style_prefs,
        render_with_paths,
        render_with_single_path,
    )
}

pub(super) fn render_before_additive_focus_with_exact_paths(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    exact_paths: &[ExprPath],
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    before::render_before_additive_focus_with_exact_paths(
        context,
        global_before_expr,
        exact_paths,
        display_hints,
        style_prefs,
        render_with_paths,
    )
}

pub(super) fn render_after_additive_focus(
    context: &Context,
    global_after_expr: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    found_paths: &[ExprPath],
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    after::render_after_additive_focus(
        context,
        global_after_expr,
        focus_after,
        found_paths,
        display_hints,
        style_prefs,
        render_with_single_path,
    )
}

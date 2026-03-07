mod focus;

use super::super::TimelineStepSnapshots;
use super::additive_render::{render_after_additive_focus, render_before_additive_focus};
use cas_ast::Context;
use cas_formatter::path::extract_add_terms;
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::Step;

pub(super) fn render_additive_focus_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    focus_before: cas_ast::ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let focus_after = step.after_local().unwrap_or(step.after);
    let focus_terms = extract_add_terms(context, focus_before);
    let found_paths = focus::collect_additive_focus_paths_with_scope(
        context,
        step,
        snapshots.global_before_expr,
        focus_before,
        &focus_terms,
    );
    let before = render_before_additive_focus(
        context,
        snapshots.global_before_expr,
        &found_paths,
        step,
        display_hints,
        style_prefs,
    );
    let after = render_after_additive_focus(
        context,
        snapshots.global_after_expr,
        focus_after,
        display_hints,
        style_prefs,
    );

    (before, after)
}

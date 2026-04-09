mod focus_path;

use super::super::renderers::render_with_single_path;
use super::additive::render_additive_focus_transition;
use super::direct::render_direct_focus_transition;
use crate::runtime::Step;
use crate::timeline::simplify_highlights::TimelineStepSnapshots;
use cas_ast::{Context, Expr, ExprId, ExprPath};
use cas_formatter::path::{diff_find_path_to_expr, diff_find_paths_by_structure};
use cas_formatter::{DisplayContext, StylePreferences};

pub(super) fn render_local_scope_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    before_local: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    if !matches!(context.get(before_local), Expr::Add(_, _) | Expr::Sub(_, _)) {
        if let Some(absolute_transition) = render_absolute_scope_transition(
            context,
            step,
            snapshots,
            before_local,
            display_hints,
            style_prefs,
        ) {
            return absolute_transition;
        }
    }

    let focus_path = focus_path::resolve_focus_path(context, step.before, before_local);

    if !focus_path.is_empty() {
        return render_direct_focus_transition(
            context,
            step,
            snapshots,
            focus_path,
            display_hints,
            style_prefs,
        );
    }

    render_additive_focus_transition(
        context,
        step,
        snapshots,
        before_local,
        display_hints,
        style_prefs,
    )
}

fn render_absolute_scope_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    before_local: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> Option<(String, String)> {
    let focus_after = step.after_local().unwrap_or(step.after);
    let before_path = find_absolute_path(context, snapshots.global_before_expr, before_local)?;
    let after_path = find_absolute_path(context, snapshots.global_after_expr, focus_after)?;

    let before = render_with_single_path(
        context,
        snapshots.global_before_expr,
        before_path,
        cas_formatter::HighlightColor::Red,
        display_hints,
        style_prefs,
    );
    let after = render_with_single_path(
        context,
        snapshots.global_after_expr,
        after_path,
        cas_formatter::HighlightColor::Green,
        display_hints,
        style_prefs,
    );
    Some((before, after))
}

fn find_absolute_path(context: &Context, root: ExprId, target: ExprId) -> Option<ExprPath> {
    diff_find_path_to_expr(context, root, target).or_else(|| {
        diff_find_paths_by_structure(context, root, target)
            .into_iter()
            .next()
    })
}

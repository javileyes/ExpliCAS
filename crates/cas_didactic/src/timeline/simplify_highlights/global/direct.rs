use super::super::renderers::render_with_single_path;
use super::super::TimelineStepSnapshots;
use crate::cas_solver::{pathsteps_to_expr_path, Step};
use cas_ast::{Context, ExprPath};
use cas_formatter::{DisplayContext, HighlightColor, StylePreferences};

pub(super) fn render_direct_focus_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    focus_path: ExprPath,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let mut extended = pathsteps_to_expr_path(step.path());
    extended.extend(focus_path);

    let before = render_with_single_path(
        context,
        snapshots.global_before_expr,
        extended.clone(),
        HighlightColor::Red,
        display_hints,
        style_prefs,
    );
    let after = render_with_single_path(
        context,
        snapshots.global_after_expr,
        extended,
        HighlightColor::Green,
        display_hints,
        style_prefs,
    );

    (before, after)
}

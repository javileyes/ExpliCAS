use super::super::renderers::render_with_single_path;
use super::super::TimelineStepSnapshots;
use crate::runtime::{pathsteps_to_expr_path, Step};
use cas_ast::Context;
use cas_formatter::{DisplayContext, HighlightColor, StylePreferences};

pub(super) fn render_default_global_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let expr_path = pathsteps_to_expr_path(step.path());
    let before = render_with_single_path(
        context,
        snapshots.global_before_expr,
        expr_path.clone(),
        HighlightColor::Red,
        display_hints,
        style_prefs,
    );
    let after = render_with_single_path(
        context,
        snapshots.global_after_expr,
        expr_path,
        HighlightColor::Green,
        display_hints,
        style_prefs,
    );
    (before, after)
}

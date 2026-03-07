mod focus_path;

use super::additive::render_additive_focus_transition;
use super::direct::render_direct_focus_transition;
use crate::timeline::simplify_highlights::TimelineStepSnapshots;
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::Step;

pub(super) fn render_local_scope_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    before_local: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
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

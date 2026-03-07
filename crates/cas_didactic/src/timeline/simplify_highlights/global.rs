mod additive;
mod additive_render;
mod additive_search;
mod default_transition;
mod direct;
mod scope;

use self::scope::render_local_scope_transition;
use super::TimelineStepSnapshots;
use cas_ast::Context;
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::Step;

pub(super) fn render_global_transition_latex(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    if let Some(before_local) = step.before_local().filter(|&bl| bl != step.before) {
        return render_local_scope_transition(
            context,
            step,
            snapshots,
            before_local,
            display_hints,
            style_prefs,
        );
    }

    default_transition::render_default_global_transition(
        context,
        step,
        snapshots,
        display_hints,
        style_prefs,
    )
}

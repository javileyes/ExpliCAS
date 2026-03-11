use super::{global, renderers, TimelineRenderedStepMath, TimelineStepSnapshots};
use crate::runtime::Step;
use cas_ast::Context;
use cas_formatter::{DisplayContext, StylePreferences};

pub(super) fn render_timeline_step_math(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> TimelineRenderedStepMath {
    let (global_before, global_after) = global::render_global_transition_latex(
        context,
        step,
        snapshots,
        display_hints,
        style_prefs,
    );
    let local_change_latex =
        renderers::render_local_change_latex(context, step, display_hints, style_prefs);

    TimelineRenderedStepMath {
        global_before,
        global_after,
        local_change_latex,
    }
}

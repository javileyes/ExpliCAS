mod global;
mod math;
mod renderers;
mod snapshots;

use crate::runtime::Step;
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TimelineStepSnapshots {
    pub global_before_expr: ExprId,
    pub global_after_expr: ExprId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TimelineRenderedStepMath {
    pub global_before: String,
    pub global_after: String,
    pub local_change_latex: String,
}

pub(crate) fn resolve_timeline_step_global_snapshots(
    context: &mut Context,
    steps: &[Step],
    original_expr: ExprId,
    step_idx: usize,
    step: &Step,
) -> TimelineStepSnapshots {
    snapshots::resolve_timeline_step_global_snapshots(context, steps, original_expr, step_idx, step)
}

pub(crate) fn render_timeline_step_math(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> TimelineRenderedStepMath {
    math::render_timeline_step_math(context, step, snapshots, display_hints, style_prefs)
}

pub(crate) fn render_step_wire_global_before_after_latex(
    context: &Context,
    step: &Step,
) -> (String, String) {
    let snapshots = TimelineStepSnapshots {
        global_before_expr: step.global_before.unwrap_or(step.before),
        global_after_expr: step.global_after.unwrap_or(step.after),
    };
    let display_hints = DisplayContext::default();
    let style_prefs = StylePreferences::default();
    global::render_global_transition_latex(context, step, snapshots, &display_hints, &style_prefs)
}

mod global;
mod math;
mod renderers;
mod snapshots;

use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::Step;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TimelineStepSnapshots {
    pub global_before_expr: ExprId,
    pub global_after_expr: ExprId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TimelineRenderedStepMath {
    pub global_before: String,
    pub global_after: String,
    pub local_change_latex: String,
}

pub(super) fn resolve_timeline_step_global_snapshots(
    context: &mut Context,
    steps: &[Step],
    original_expr: ExprId,
    step_idx: usize,
    step: &Step,
) -> TimelineStepSnapshots {
    snapshots::resolve_timeline_step_global_snapshots(context, steps, original_expr, step_idx, step)
}

pub(super) fn render_timeline_step_math(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> TimelineRenderedStepMath {
    math::render_timeline_step_math(context, step, snapshots, display_hints, style_prefs)
}

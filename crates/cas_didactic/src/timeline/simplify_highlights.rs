mod global;
mod renderers;

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
    let global_before_expr = step.global_before.unwrap_or_else(|| {
        if step_idx == 0 {
            original_expr
        } else {
            steps
                .get(step_idx - 1)
                .and_then(|prev| prev.global_after)
                .unwrap_or(original_expr)
        }
    });
    let global_after_expr = step.global_after.unwrap_or_else(|| {
        cas_solver::reconstruct_global_expr(context, global_before_expr, step.path(), step.after)
    });

    TimelineStepSnapshots {
        global_before_expr,
        global_after_expr,
    }
}

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

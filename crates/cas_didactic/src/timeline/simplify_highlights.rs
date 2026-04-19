mod global;
mod math;
mod renderers;
mod snapshots;

use crate::runtime::Step;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};
use num_traits::Zero;

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
    let mut temp_ctx = context.clone();
    let snapshots = step_wire_presentation_snapshots(&mut temp_ctx, step);
    let mut presentation_step = step.clone();
    presentation_step.before =
        cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut temp_ctx, step.before);
    presentation_step.after =
        cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut temp_ctx, step.after);
    if let Some(meta) = presentation_step.meta.as_mut() {
        if let Some(before_local) = meta.before_local {
            meta.before_local = Some(
                cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
                    &mut temp_ctx,
                    before_local,
                ),
            );
        }
        if let Some(after_local) = meta.after_local {
            meta.after_local = Some(
                cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
                    &mut temp_ctx,
                    after_local,
                ),
            );
        }
    }
    let display_hints = DisplayContext::default();
    let style_prefs = StylePreferences::default();
    global::render_global_transition_latex(
        &temp_ctx,
        &presentation_step,
        snapshots,
        &display_hints,
        &style_prefs,
    )
}

pub(crate) fn step_wire_presentation_snapshots(
    context: &mut Context,
    step: &Step,
) -> TimelineStepSnapshots {
    let raw_global_before_expr = step.global_before.unwrap_or(step.before);
    let raw_global_after_expr = step.global_after.unwrap_or(step.after);
    let global_before_expr = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
        context,
        raw_global_before_expr,
    );
    let global_after_expr = cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
        context,
        raw_global_after_expr,
    );

    if is_zero_expr(context, global_after_expr) && step.before != raw_global_before_expr {
        return TimelineStepSnapshots {
            global_before_expr: cas_solver_core::eval_step_pipeline::normalize_expr_for_display(
                context,
                step.before_local().unwrap_or(step.before),
            ),
            global_after_expr,
        };
    }

    TimelineStepSnapshots {
        global_before_expr,
        global_after_expr,
    }
}

fn is_zero_expr(context: &Context, expr: ExprId) -> bool {
    matches!(context.get(expr), Expr::Number(n) if n.is_zero())
}

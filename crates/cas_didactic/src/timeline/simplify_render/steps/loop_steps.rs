use super::super::super::simplify_highlights::TimelineStepSnapshots;
use super::super::super::simplify_substeps::TimelineSubstepsRenderState;
use crate::runtime::Step;
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};
use std::collections::HashSet;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn render_timeline_filtered_steps(
    context: &mut Context,
    html: &mut String,
    steps: &[Step],
    original_expr: ExprId,
    style_prefs: &StylePreferences,
    enriched_steps: &[crate::didactic::EnrichedStep],
    display_hints: &DisplayContext,
    filtered_indices: &HashSet<*const Step>,
    substeps_state: &mut TimelineSubstepsRenderState,
    resolve_timeline_step_global_snapshots: fn(
        &mut Context,
        &[Step],
        ExprId,
        usize,
        &Step,
    ) -> TimelineStepSnapshots,
    render_timeline_step_block: fn(
        &mut Context,
        usize,
        &Step,
        TimelineStepSnapshots,
        &StylePreferences,
        Option<&crate::didactic::EnrichedStep>,
        &mut TimelineSubstepsRenderState,
        &DisplayContext,
    ) -> String,
) -> ExprId {
    let mut step_number = 0;
    let mut last_global_after = original_expr;

    for (step_idx, step) in steps.iter().enumerate() {
        let snapshots =
            resolve_timeline_step_global_snapshots(context, steps, original_expr, step_idx, step);
        last_global_after = snapshots.global_after_expr;

        if !filtered_indices.contains(&(step as *const Step)) {
            continue;
        }
        step_number += 1;

        html.push_str(&render_timeline_step_block(
            context,
            step_number,
            step,
            snapshots,
            style_prefs,
            enriched_steps.get(step_idx),
            substeps_state,
            display_hints,
        ));
    }

    last_global_after
}

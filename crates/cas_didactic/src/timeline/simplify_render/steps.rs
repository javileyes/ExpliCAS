mod render_step;

use super::{resolve_timeline_step_global_snapshots, TimelineSubstepsRenderState};
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::Step;
use std::collections::HashSet;

pub(super) fn render_timeline_filtered_steps(
    context: &mut Context,
    html: &mut String,
    steps: &[Step],
    original_expr: ExprId,
    style_prefs: &StylePreferences,
    enriched_steps: &[crate::didactic::EnrichedStep],
    display_hints: &DisplayContext,
    filtered_indices: &HashSet<*const Step>,
) -> ExprId {
    let mut step_number = 0;
    let mut last_global_after = original_expr;
    let mut substeps_state = TimelineSubstepsRenderState::default();

    for (step_idx, step) in steps.iter().enumerate() {
        let snapshots =
            resolve_timeline_step_global_snapshots(context, steps, original_expr, step_idx, step);
        last_global_after = snapshots.global_after_expr;

        if !filtered_indices.contains(&(step as *const Step)) {
            continue;
        }
        step_number += 1;

        html.push_str(&render_step::render_timeline_step_block(
            context,
            step_number,
            step,
            snapshots,
            style_prefs,
            enriched_steps.get(step_idx),
            &mut substeps_state,
            display_hints,
        ));
    }

    last_global_after
}

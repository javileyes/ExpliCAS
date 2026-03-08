mod loop_steps;
mod render_step;

use super::super::simplify_highlights::resolve_timeline_step_global_snapshots;
use super::super::simplify_substeps::TimelineSubstepsRenderState;
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::Step;
use std::collections::HashSet;

#[allow(clippy::too_many_arguments)]
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
    let mut substeps_state = TimelineSubstepsRenderState::default();

    loop_steps::render_timeline_filtered_steps(
        context,
        html,
        steps,
        original_expr,
        style_prefs,
        enriched_steps,
        display_hints,
        filtered_indices,
        &mut substeps_state,
        resolve_timeline_step_global_snapshots,
        render_step::render_timeline_step_block,
    )
}

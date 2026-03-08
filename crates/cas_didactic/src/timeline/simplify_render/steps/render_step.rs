use super::super::super::simplify_highlights::render_timeline_step_math;
use super::super::super::simplify_highlights::TimelineStepSnapshots;
use super::super::super::simplify_step_html::render_timeline_step_html;
use super::super::super::simplify_substeps::{
    render_timeline_domain_assumptions_html, render_timeline_enriched_substeps_html,
    render_timeline_rule_substeps_html, TimelineSubstepsRenderState,
};
use cas_ast::Context;
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::Step;

#[allow(clippy::too_many_arguments)]
pub(super) fn render_timeline_step_block(
    context: &mut Context,
    step_number: usize,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    style_prefs: &StylePreferences,
    enriched_step: Option<&crate::didactic::EnrichedStep>,
    substeps_state: &mut TimelineSubstepsRenderState,
    display_hints: &DisplayContext,
) -> String {
    let rendered_step_math =
        render_timeline_step_math(context, step, snapshots, display_hints, style_prefs);
    let sub_steps_html = enriched_step
        .map(|enriched| render_timeline_enriched_substeps_html(enriched, substeps_state))
        .unwrap_or_default();

    render_timeline_step_html(
        step_number,
        step,
        &rendered_step_math,
        &sub_steps_html,
        &render_timeline_rule_substeps_html(step),
        &render_timeline_domain_assumptions_html(step),
    )
}

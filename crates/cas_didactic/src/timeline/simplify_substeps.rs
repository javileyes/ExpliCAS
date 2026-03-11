mod domain;
mod enriched;
mod rule;

use crate::cas_solver::Step;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct TimelineSubstepsRenderState {
    enriched_dedupe_shown: bool,
}

pub(super) fn render_timeline_enriched_substeps_html(
    enriched: &crate::didactic::EnrichedStep,
    state: &mut TimelineSubstepsRenderState,
) -> String {
    enriched::render_timeline_enriched_substeps_html(enriched, state)
}

pub(super) fn render_timeline_rule_substeps_html(step: &Step) -> String {
    rule::render_timeline_rule_substeps_html(step)
}

pub(super) fn render_timeline_domain_assumptions_html(step: &Step) -> String {
    domain::render_timeline_domain_assumptions_html(step)
}

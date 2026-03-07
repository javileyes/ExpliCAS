use super::super::TimelineSubstepsRenderState;

pub(super) fn should_render_enriched_substeps(
    enriched: &crate::didactic::EnrichedStep,
    state: &TimelineSubstepsRenderState,
) -> bool {
    let classification = crate::didactic::classify_sub_steps(&enriched.sub_steps);
    if classification.has_nested_fraction || classification.has_factorization {
        true
    } else {
        !state.enriched_dedupe_shown
    }
}

pub(super) fn update_enriched_substeps_state(
    render_plan: &crate::didactic::TimelineSubstepsRenderPlan,
    state: &mut TimelineSubstepsRenderState,
) {
    if render_plan.dedupe_once {
        state.enriched_dedupe_shown = true;
    }
}

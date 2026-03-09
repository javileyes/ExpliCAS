use cas_api_models::SubStepJson;
use cas_solver::Step;

pub(super) fn collect_step_json_substeps(
    step: &Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Vec<SubStepJson> {
    crate::step_payload_render::collect_step_payload_substeps(step, enriched)
}

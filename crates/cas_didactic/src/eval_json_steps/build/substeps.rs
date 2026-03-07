use cas_api_models::SubStepJson;
use cas_solver::Step;

pub(super) fn collect_step_json_substeps(
    step: &Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Vec<SubStepJson> {
    crate::eval_json_render::collect_step_json_substeps(step, enriched)
}

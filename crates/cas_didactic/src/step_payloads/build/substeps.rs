use crate::cas_solver::Step;
use cas_api_models::SubStepWire;

pub(super) fn collect_step_wire_substeps(
    step: &Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Vec<SubStepWire> {
    crate::step_payload_render::collect_step_payload_substeps(step, enriched)
}

use super::super::payload::EvalJsonResultPayload;
use crate::eval_json_finalize_input::EvalJsonFinalizeShared;
use crate::eval_json_finalize_wire::build_eval_wire_value;

pub(super) fn build_eval_json_wire(
    payload: &EvalJsonResultPayload,
    steps_count: usize,
    shared: &EvalJsonFinalizeShared<'_>,
) -> Option<serde_json::Value> {
    build_eval_wire_value(
        &shared.warnings,
        &shared.required_display,
        &payload.result,
        payload.result_latex.as_deref(),
        steps_count,
        shared.steps_mode,
    )
}

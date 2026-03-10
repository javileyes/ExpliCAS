use super::super::payload::EvalOutputResultPayload;
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;
use crate::eval_output_finalize_wire::build_eval_output_wire_value;

pub(super) fn build_eval_output_wire(
    payload: &EvalOutputResultPayload,
    steps_count: usize,
    shared: &EvalOutputFinalizeShared<'_>,
) -> Option<serde_json::Value> {
    build_eval_output_wire_value(
        &shared.warnings,
        &shared.required_display,
        &payload.result,
        payload.result_latex.as_deref(),
        steps_count,
        shared.steps_mode,
    )
}

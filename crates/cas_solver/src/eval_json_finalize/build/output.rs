mod build;
mod wire;

use cas_api_models::EvalJsonOutput;

use super::payload::EvalJsonResultPayload;
use crate::eval_json_finalize_input::EvalJsonFinalizeShared;

pub(crate) fn build_eval_json_output(
    payload: EvalJsonResultPayload,
    steps_count: usize,
    shared: EvalJsonFinalizeShared<'_>,
) -> EvalJsonOutput {
    let wire = wire::build_eval_json_wire(&payload, steps_count, &shared);
    build::build_eval_json_output(payload, steps_count, shared, wire)
}

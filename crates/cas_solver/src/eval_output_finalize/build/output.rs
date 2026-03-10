mod build;
mod wire;

use cas_api_models::EvalJsonOutput;

use super::payload::EvalOutputResultPayload;
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;

pub(crate) fn build_eval_output(
    payload: EvalOutputResultPayload,
    steps_count: usize,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalJsonOutput {
    let wire = wire::build_eval_output_wire(&payload, steps_count, &shared);
    build::build_eval_output(payload, steps_count, shared, wire)
}

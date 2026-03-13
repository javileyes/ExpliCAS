#[path = "output/build.rs"]
mod build;
#[path = "output/wire.rs"]
mod wire;

use super::payload::EvalOutputResultPayload;
use crate::eval_output_finalize::EvalOutputWire;
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;

pub(crate) fn build_eval_output(
    payload: EvalOutputResultPayload,
    steps_count: usize,
    shared: EvalOutputFinalizeShared<'_>,
) -> EvalOutputWire {
    let wire = wire::build_eval_output_wire(&payload, steps_count, &shared);
    build::build_eval_output(payload, steps_count, shared, wire)
}

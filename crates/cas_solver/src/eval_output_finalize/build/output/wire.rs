use super::super::payload::EvalOutputResultPayload;
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;
use crate::eval_output_finalize_wire::build_eval_output_wire_value;
use cas_api_models::wire::EvalWireReplyParts;

pub(super) fn build_eval_output_wire(
    payload: &EvalOutputResultPayload,
    steps_count: usize,
    shared: &EvalOutputFinalizeShared<'_>,
) -> Option<serde_json::Value> {
    build_eval_output_wire_value(EvalWireReplyParts {
        warnings: &shared.warnings,
        assumptions_used: &shared.assumptions_used,
        required_display: &shared.required_display,
        blocked_hints: &shared.blocked_hints,
        strategy: shared.strategy.as_deref(),
        result: &payload.result,
        result_latex: payload.result_latex.as_deref(),
        steps_count,
        steps_mode: shared.steps_mode,
    })
}

use cas_api_models::wire::{build_eval_wire_reply, EvalWireReplyParts};

pub(crate) fn build_eval_output_wire_value(
    parts: EvalWireReplyParts<'_>,
) -> Option<serde_json::Value> {
    serde_json::to_value(build_eval_wire_reply(parts)).ok()
}

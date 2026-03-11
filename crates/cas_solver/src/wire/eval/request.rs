use crate::eval_input::{build_prepared_eval_request_for_input, PreparedEvalRequest};
use cas_api_models::{EngineWireError, SpanWire};

pub(super) fn build_prepared_eval_request(
    expr: &str,
    engine: &mut crate::Engine,
) -> Result<PreparedEvalRequest, EngineWireError> {
    build_prepared_eval_request_for_input(expr, &mut engine.simplifier.context, false)
        .map_err(|e| EngineWireError::parse(e, Option::<SpanWire>::None))
}

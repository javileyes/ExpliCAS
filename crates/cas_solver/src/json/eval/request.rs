use cas_api_models::{EngineJsonError, SpanJson};

pub(super) fn build_prepared_eval_json_request(
    expr: &str,
    engine: &mut crate::Engine,
) -> Result<crate::EvalJsonPreparedRequest, EngineJsonError> {
    crate::build_eval_json_request_for_input(expr, &mut engine.simplifier.context, false)
        .map_err(|e| EngineJsonError::parse(e, Option::<SpanJson>::None))
}

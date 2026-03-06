use crate::eval_json_input::EvalJsonPreparedRequest;

/// Evaluate one prepared eval-json request in stateless mode with provided options.
pub(super) fn evaluate_prepared_stateless_request(
    engine: &mut crate::Engine,
    options: crate::EvalOptions,
    prepared: EvalJsonPreparedRequest,
) -> Result<crate::EvalOutputView, String> {
    let mut session = crate::StatelessEvalSession::new(options);
    crate::eval_json_request_runtime::evaluate_prepared_request_with_session(
        engine,
        &mut session,
        prepared,
    )
}

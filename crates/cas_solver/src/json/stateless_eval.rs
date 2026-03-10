use crate::eval_input::PreparedEvalRequest;

/// Evaluate one prepared eval-json request in stateless mode with provided options.
pub(super) fn evaluate_prepared_stateless_request(
    engine: &mut crate::Engine,
    options: crate::EvalOptions,
    prepared: PreparedEvalRequest,
) -> Result<crate::EvalOutputView, String> {
    let mut session = crate::StatelessEvalSession::new(options);
    crate::eval_request_runtime::evaluate_prepared_request_with_session(
        engine,
        &mut session,
        prepared,
    )
}

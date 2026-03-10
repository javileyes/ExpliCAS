use crate::eval_input::{EvalNonSolveAction, PreparedEvalRequest};

fn map_non_solve_action(action: EvalNonSolveAction) -> crate::EvalAction {
    match action {
        EvalNonSolveAction::Simplify => crate::EvalAction::Simplify,
        EvalNonSolveAction::Limit { var, approach } => crate::EvalAction::Limit { var, approach },
    }
}

/// Evaluate one prepared eval-json request with any eval session implementation.
pub(crate) fn evaluate_prepared_request_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    prepared: PreparedEvalRequest,
) -> Result<crate::EvalOutputView, String>
where
    S: crate::SolverEvalSession,
{
    match prepared {
        PreparedEvalRequest::Solve {
            raw_input,
            parsed,
            var,
            auto_store,
        } => crate::solve_command_eval_core::evaluate_solve_parsed_with_session(
            &mut engine.simplifier,
            session,
            raw_input,
            parsed,
            &var,
            auto_store,
        ),
        PreparedEvalRequest::Eval {
            raw_input,
            parsed,
            action,
            auto_store,
        } => {
            let req = crate::EvalRequest {
                raw_input,
                parsed,
                action: map_non_solve_action(action),
                auto_store,
            };
            let output = engine.eval(session, req).map_err(|e| e.to_string())?;
            Ok(crate::eval_output_view(&output))
        }
    }
}

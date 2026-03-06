use crate::eval_json_input::{EvalJsonNonSolveAction, EvalJsonPreparedRequest};

fn map_non_solve_action(action: EvalJsonNonSolveAction) -> crate::EvalAction {
    match action {
        EvalJsonNonSolveAction::Simplify => crate::EvalAction::Simplify,
        EvalJsonNonSolveAction::Limit { var, approach } => {
            crate::EvalAction::Limit { var, approach }
        }
    }
}

/// Evaluate one prepared eval-json request with any eval session implementation.
pub(crate) fn evaluate_prepared_request_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    prepared: EvalJsonPreparedRequest,
) -> Result<crate::EvalOutputView, String>
where
    S: crate::SolverEvalSession,
{
    match prepared {
        EvalJsonPreparedRequest::Solve {
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
        EvalJsonPreparedRequest::Eval {
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

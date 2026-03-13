use crate::eval_input_variable::detect_solve_variable_for_eval_request;

use super::super::{EvalNonSolveAction, PreparedEvalRequest};

pub(super) fn build_statement_request(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
) -> Result<PreparedEvalRequest, String> {
    let stmt = crate::parse_statement_or_session_ref(ctx, raw_input)
        .map_err(|e| format!("Parse error: {e}"))?;
    match stmt {
        cas_parser::Statement::Equation(eq) => {
            let parsed = ctx.call("Equal", vec![eq.lhs, eq.rhs]);
            let var = detect_solve_variable_for_eval_request(ctx, eq.lhs, eq.rhs);
            Ok(PreparedEvalRequest::Solve {
                raw_input: raw_input.to_string(),
                parsed,
                var,
                auto_store,
            })
        }
        cas_parser::Statement::Expression(parsed) => Ok(PreparedEvalRequest::Eval {
            raw_input: raw_input.to_string(),
            parsed,
            action: EvalNonSolveAction::Simplify,
            auto_store,
        }),
    }
}

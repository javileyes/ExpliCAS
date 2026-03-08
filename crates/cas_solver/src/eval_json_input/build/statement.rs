use crate::eval_json_input_variable::detect_solve_variable_eval_json;

use super::super::types::{EvalJsonNonSolveAction, EvalJsonPreparedRequest};

pub(super) fn build_statement_request(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
) -> Result<EvalJsonPreparedRequest, String> {
    let stmt = crate::parse_statement_or_session_ref(ctx, raw_input)
        .map_err(|e| format!("Parse error: {e}"))?;
    match stmt {
        cas_parser::Statement::Equation(eq) => {
            let parsed = ctx.call("Equal", vec![eq.lhs, eq.rhs]);
            let var = detect_solve_variable_eval_json(ctx, eq.lhs, eq.rhs);
            Ok(EvalJsonPreparedRequest::Solve {
                raw_input: raw_input.to_string(),
                parsed,
                var,
                auto_store,
            })
        }
        cas_parser::Statement::Expression(parsed) => Ok(EvalJsonPreparedRequest::Eval {
            raw_input: raw_input.to_string(),
            parsed,
            action: EvalJsonNonSolveAction::Simplify,
            auto_store,
        }),
    }
}

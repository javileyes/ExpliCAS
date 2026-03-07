use crate::eval_json_input_special::{map_limit_approach, parse_solve_input_as_equation_expr};
use crate::eval_json_input_variable::detect_solve_variable_eval_json;
use cas_api_models::{parse_eval_json_special_command, EvalJsonSpecialCommand};

use super::types::{EvalJsonNonSolveAction, EvalJsonPreparedRequest};

/// Build eval-json request as a solver-owned action enum.
pub fn build_eval_json_request_for_input(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
) -> Result<EvalJsonPreparedRequest, String> {
    if let Some(command) = parse_eval_json_special_command(raw_input) {
        return match command {
            EvalJsonSpecialCommand::Solve { equation, var } => {
                let parsed = parse_solve_input_as_equation_expr(ctx, &equation)
                    .map_err(|e| format!("Parse error in solve equation: {e}"))?;
                Ok(EvalJsonPreparedRequest::Solve {
                    raw_input: raw_input.to_string(),
                    parsed,
                    var,
                    auto_store,
                })
            }
            EvalJsonSpecialCommand::Limit {
                expr,
                var,
                approach,
            } => {
                let parsed = cas_parser::parse(&expr, ctx)
                    .map_err(|e| format!("Parse error in limit expression: {e}"))?;
                Ok(EvalJsonPreparedRequest::Eval {
                    raw_input: raw_input.to_string(),
                    parsed,
                    action: EvalJsonNonSolveAction::Limit {
                        var,
                        approach: map_limit_approach(approach),
                    },
                    auto_store,
                })
            }
        };
    }

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

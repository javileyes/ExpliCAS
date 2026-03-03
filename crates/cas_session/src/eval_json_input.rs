//! Input parsing and request-building helpers for eval-json orchestration.

use crate::eval_json_input_special::{map_limit_approach, parse_solve_input_as_equation_expr};
use crate::eval_json_input_variable::detect_solve_variable_eval_json;
use cas_api_models::{parse_eval_json_special_command, EvalJsonSpecialCommand};

/// Build canonical eval request from raw eval-json input.
pub(crate) fn build_eval_request_for_input(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
) -> Result<cas_solver::EvalRequest, String> {
    if let Some(command) = parse_eval_json_special_command(raw_input) {
        return match command {
            EvalJsonSpecialCommand::Solve { equation, var } => {
                let parsed = parse_solve_input_as_equation_expr(ctx, &equation)
                    .map_err(|e| format!("Parse error in solve equation: {e}"))?;

                Ok(cas_solver::EvalRequest {
                    raw_input: raw_input.to_string(),
                    parsed,
                    action: cas_solver::EvalAction::Solve { var },
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

                Ok(cas_solver::EvalRequest {
                    raw_input: raw_input.to_string(),
                    parsed,
                    action: cas_solver::EvalAction::Limit {
                        var,
                        approach: map_limit_approach(approach),
                    },
                    auto_store,
                })
            }
        };
    }

    let stmt = crate::input_parse_common::parse_statement_or_session_ref(ctx, raw_input)
        .map_err(|e| format!("Parse error: {e}"))?;
    match stmt {
        cas_parser::Statement::Equation(eq) => {
            let parsed = ctx.call("Equal", vec![eq.lhs, eq.rhs]);
            let var = detect_solve_variable_eval_json(ctx, eq.lhs, eq.rhs);
            Ok(cas_solver::EvalRequest {
                raw_input: raw_input.to_string(),
                parsed,
                action: cas_solver::EvalAction::Solve { var },
                auto_store,
            })
        }
        cas_parser::Statement::Expression(parsed) => Ok(cas_solver::EvalRequest {
            raw_input: raw_input.to_string(),
            parsed,
            action: cas_solver::EvalAction::Simplify,
            auto_store,
        }),
    }
}

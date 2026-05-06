use crate::eval_input_special::{map_limit_approach, parse_solve_input_for_eval_request};
use cas_api_models::EvalSpecialCommand;

use super::super::{EvalNonSolveAction, PreparedEvalRequest};

pub(super) fn build_special_command_request(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
    command: EvalSpecialCommand,
) -> Result<PreparedEvalRequest, String> {
    match command {
        EvalSpecialCommand::Solve { equation, var } => {
            let (parsed, original_equation) = parse_solve_input_for_eval_request(ctx, &equation)
                .map_err(|e| format!("Parse error in solve equation: {e}"))?;
            Ok(PreparedEvalRequest::Solve {
                raw_input: raw_input.to_string(),
                parsed,
                original_equation,
                var,
                auto_store,
            })
        }
        EvalSpecialCommand::SolveSystem { input } => {
            let spec = crate::linear_system_command_parse::parse_linear_system_spec(ctx, &input)
                .map_err(|error| {
                    crate::linear_system_command_format::format_linear_system_command_error_message(
                        &crate::linear_system_command_eval::LinearSystemCommandEvalError::Parse(
                            error,
                        ),
                    )
                })?;
            let parsed_anchor = spec.exprs.first().copied().ok_or_else(|| {
                "Internal error: solve_system parsed without equations".to_string()
            })?;
            Ok(PreparedEvalRequest::SolveSystem {
                parsed_anchor,
                exprs: spec.exprs,
                vars: spec.vars,
            })
        }
        EvalSpecialCommand::Derive { input } => {
            let (parsed, target) = crate::parse_expr_pair(ctx, &input)
                .map_err(|e| crate::format_expr_pair_parse_error_message(&e, "derive"))?;
            Ok(PreparedEvalRequest::Derive {
                raw_input: raw_input.to_string(),
                parsed,
                target,
                auto_store,
            })
        }
        EvalSpecialCommand::Equiv { input } => {
            let (parsed, other) = crate::parse_expr_pair(ctx, &input)
                .map_err(|e| crate::format_expr_pair_parse_error_message(&e, "equiv"))?;
            Ok(PreparedEvalRequest::Eval {
                raw_input: raw_input.to_string(),
                parsed,
                action: EvalNonSolveAction::Equiv { other },
                auto_store,
            })
        }
        EvalSpecialCommand::Limit {
            expr,
            var,
            approach,
        } => {
            let parsed = cas_parser::parse(&expr, ctx)
                .map_err(|e| format!("Parse error in limit expression: {e}"))?;
            Ok(PreparedEvalRequest::Eval {
                raw_input: raw_input.to_string(),
                parsed,
                action: EvalNonSolveAction::Limit {
                    var,
                    approach: map_limit_approach(ctx, approach)?,
                },
                auto_store,
            })
        }
    }
}

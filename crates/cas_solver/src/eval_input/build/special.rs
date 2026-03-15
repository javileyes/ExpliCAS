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
                    approach: map_limit_approach(approach),
                },
                auto_store,
            })
        }
    }
}

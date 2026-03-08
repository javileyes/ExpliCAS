use crate::eval_json_input_special::{map_limit_approach, parse_solve_input_as_equation_expr};
use cas_api_models::EvalJsonSpecialCommand;

use super::super::types::{EvalJsonNonSolveAction, EvalJsonPreparedRequest};

pub(super) fn build_special_command_request(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
    command: EvalJsonSpecialCommand,
) -> Result<EvalJsonPreparedRequest, String> {
    match command {
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
    }
}

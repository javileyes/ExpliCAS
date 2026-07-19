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
        EvalSpecialCommand::Dsolve {
            equation,
            func,
            var,
            conditions,
        } => {
            // The ODE equation parses to an `Equal(lhs, rhs)` tree that the
            // dsolve action reads RAW (never pre-simplified: `diff(y,x)` would
            // collapse to `0`).
            let (parsed, _original_equation) =
                parse_solve_input_for_eval_request(ctx, &equation)
                    .map_err(|e| format!("Parse error in dsolve equation: {e}"))?;
            let conditions = parse_dsolve_conditions(ctx, &conditions, &func)?;
            Ok(PreparedEvalRequest::Eval {
                raw_input: raw_input.to_string(),
                parsed,
                action: EvalNonSolveAction::Dsolve {
                    func,
                    var,
                    conditions,
                },
                auto_store,
            })
        }
    }
}

/// Split each textual dsolve condition (`y(0)=3`, `y'(0)=2`) and parse point
/// and value separately — the head never reaches the expression parser (D1).
pub(crate) fn parse_dsolve_conditions(
    ctx: &mut cas_ast::Context,
    conditions: &[String],
    func: &str,
) -> Result<Vec<cas_solver_core::eval_models::DsolveCondition>, String> {
    let mut parsed = Vec::with_capacity(conditions.len());
    for cond_text in conditions {
        let Some((point_text, value_text, order)) =
            cas_api_models::split_dsolve_initial_condition(cond_text, func)
        else {
            return Err(format!(
                "Invalid dsolve initial condition `{cond_text}`: expected {func}(x0) = y0, e.g. {func}(0) = 3."
            ));
        };
        let point = cas_parser::parse(&point_text, ctx)
            .map_err(|e| format!("Parse error in dsolve condition point: {e}"))?;
        let value = cas_parser::parse(&value_text, ctx)
            .map_err(|e| format!("Parse error in dsolve condition value: {e}"))?;
        parsed.push(cas_solver_core::eval_models::DsolveCondition {
            point,
            value,
            order,
        });
    }
    Ok(parsed)
}

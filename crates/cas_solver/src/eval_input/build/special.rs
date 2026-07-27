use crate::eval_input_special::{map_limit_approach, parse_solve_input_for_eval_request};
use cas_api_models::EvalSpecialCommand;

use super::super::{EvalNonSolveAction, PreparedEvalRequest};

/// Canonicalize a Greek-glyph var/func token to its spelled name (`λ` →
/// `lambda`). The EXPRESSION side of every command goes through cas_parser,
/// which does this aliasing at the identifier level — but var/func names
/// travel the command wire as RAW strings and are later looked up by name.
/// Without this, `solve(λ^2-4=0, λ)` interns `lambda` from the equation and
/// then searches for a variable literally named 'λ'.
fn canonical_var_token(name: String) -> String {
    let canonical = cas_ast::canonical_greek_token(&name);
    if canonical == name {
        name
    } else {
        canonical.to_string()
    }
}

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
                var: canonical_var_token(var),
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
                    var: canonical_var_token(var),
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
            // Conditions match their head against what the user TYPED (θ(0)=3
            // has head θ), so split with the raw func; the action then gets
            // the canonical name, matching the parsed equation's symbols.
            let conditions = parse_dsolve_conditions(ctx, &conditions, &func)?;
            Ok(PreparedEvalRequest::Eval {
                raw_input: raw_input.to_string(),
                parsed,
                action: EvalNonSolveAction::Dsolve {
                    func: canonical_var_token(func),
                    var: canonical_var_token(var),
                    conditions,
                },
                auto_store,
            })
        }
        EvalSpecialCommand::DsolveSystem {
            equations,
            funcs,
            var,
            conditions,
        } => {
            let (parsed, _) = parse_solve_input_for_eval_request(ctx, &equations[0])
                .map_err(|e| format!("Parse error in dsolve system equation 1: {e}"))?;
            let (second_equation, _) = parse_solve_input_for_eval_request(ctx, &equations[1])
                .map_err(|e| format!("Parse error in dsolve system equation 2: {e}"))?;
            // Conditions on either unknown parse against whichever head matches.
            let mut parsed_conditions = Vec::new();
            for cond_text in &conditions {
                let mut matched = None;
                for f in &funcs {
                    if let Some(parts) =
                        cas_api_models::split_dsolve_initial_condition(cond_text, f)
                    {
                        matched = Some(parts);
                        break;
                    }
                }
                let Some((point_text, value_text, order)) = matched else {
                    return Err(format!(
                        "Invalid dsolve system initial condition `{cond_text}`."
                    ));
                };
                let point = cas_parser::parse(&point_text, ctx)
                    .map_err(|e| format!("Parse error in dsolve condition point: {e}"))?;
                let value = cas_parser::parse(&value_text, ctx)
                    .map_err(|e| format!("Parse error in dsolve condition value: {e}"))?;
                parsed_conditions.push(cas_solver_core::eval_models::DsolveCondition {
                    point,
                    value,
                    order,
                });
            }
            Ok(PreparedEvalRequest::Eval {
                raw_input: raw_input.to_string(),
                parsed,
                action: EvalNonSolveAction::DsolveSystem {
                    second_equation,
                    funcs: funcs.into_iter().map(canonical_var_token).collect(),
                    var: canonical_var_token(var),
                    conditions: parsed_conditions,
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

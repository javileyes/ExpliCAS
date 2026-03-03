//! Input parsing and request-building helpers for eval-json orchestration.

use cas_api_models::{
    parse_eval_json_special_command, EvalJsonLimitApproach, EvalJsonSpecialCommand,
};
use cas_ast::{Context, Expr, ExprId};

fn preferred_variable_fallback(vars: &[String]) -> String {
    if vars.iter().any(|v| v == "x") {
        return "x".to_string();
    }

    for preferred in ["y", "z", "t", "n", "a", "b", "c"] {
        if vars.iter().any(|v| v == preferred) {
            return preferred.to_string();
        }
    }

    vars.first().cloned().unwrap_or_else(|| "x".to_string())
}

fn detect_solve_variable_eval_json(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> String {
    let equation_residual = ctx.add(Expr::Sub(lhs, rhs));

    match cas_solver::infer_solve_variable(ctx, equation_residual) {
        Ok(Some(var)) => var,
        Ok(None) => "x".to_string(),
        Err(vars) => preferred_variable_fallback(&vars),
    }
}

fn map_limit_approach(approach: EvalJsonLimitApproach) -> cas_solver::Approach {
    match approach {
        EvalJsonLimitApproach::PosInfinity => cas_solver::Approach::PosInfinity,
        EvalJsonLimitApproach::NegInfinity => cas_solver::Approach::NegInfinity,
    }
}

fn parse_solve_input_as_equation_expr(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<ExprId, String> {
    let stmt = crate::input_parse_common::parse_statement_or_session_ref(ctx, input)?;
    let parsed = match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => {
            let zero = ctx.num(0);
            ctx.call("Equal", vec![expr, zero])
        }
    };
    Ok(parsed)
}

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

#[cfg(test)]
mod tests {
    use super::detect_solve_variable_eval_json;

    #[test]
    fn detect_solve_variable_prefers_x_when_ambiguous() {
        let mut ctx = cas_ast::Context::new();
        let statement = cas_parser::parse_statement("x + y = 0", &mut ctx).expect("parse");
        let (lhs, rhs) = match statement {
            cas_parser::Statement::Equation(eq) => (eq.lhs, eq.rhs),
            _ => panic!("expected equation"),
        };
        let var = detect_solve_variable_eval_json(&mut ctx, lhs, rhs);
        assert_eq!(var, "x");
    }

    #[test]
    fn detect_solve_variable_falls_back_to_preferred_order() {
        let mut ctx = cas_ast::Context::new();
        let statement = cas_parser::parse_statement("z + y = 0", &mut ctx).expect("parse");
        let (lhs, rhs) = match statement {
            cas_parser::Statement::Equation(eq) => (eq.lhs, eq.rhs),
            _ => panic!("expected equation"),
        };
        let var = detect_solve_variable_eval_json(&mut ctx, lhs, rhs);
        assert_eq!(var, "y");
    }
}

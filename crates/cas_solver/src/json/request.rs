use crate::{EvalAction, EvalRequest};
use cas_ast::ExprId;
use cas_formatter::LaTeXExpr;

/// Build an eval request from eval-json raw input.
///
/// Preserves current CLI behavior:
/// - `solve(...)` / `limit(...)` special forms
/// - plain equation auto-detected as solve with inferred variable
/// - plain expression as simplify
pub fn build_eval_request_for_input(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
) -> Result<EvalRequest, String> {
    if let Some(command) = crate::json::parse_eval_json_special_command(raw_input) {
        return match command {
            crate::json::EvalJsonSpecialCommand::Solve { equation, var } => {
                let parsed = parse_solve_input_as_equation_expr(ctx, &equation)
                    .map_err(|e| format!("Parse error in solve equation: {}", e))?;

                Ok(EvalRequest {
                    raw_input: raw_input.to_string(),
                    parsed,
                    action: EvalAction::Solve { var },
                    auto_store,
                })
            }
            crate::json::EvalJsonSpecialCommand::Limit {
                expr,
                var,
                approach,
            } => {
                let parsed = cas_parser::parse(&expr, ctx)
                    .map_err(|e| format!("Parse error in limit expression: {}", e))?;

                Ok(EvalRequest {
                    raw_input: raw_input.to_string(),
                    parsed,
                    action: EvalAction::Limit { var, approach },
                    auto_store,
                })
            }
        };
    }

    let stmt = crate::parse_statement_or_session_ref(ctx, raw_input)
        .map_err(|e| format!("Parse error: {}", e))?;
    match stmt {
        cas_parser::Statement::Equation(eq) => {
            let parsed = ctx.call("Equal", vec![eq.lhs, eq.rhs]);
            let var = crate::json::detect_solve_variable_eval_json(ctx, eq.lhs, eq.rhs);
            Ok(EvalRequest {
                raw_input: raw_input.to_string(),
                parsed,
                action: EvalAction::Solve { var },
                auto_store,
            })
        }
        cas_parser::Statement::Expression(parsed) => Ok(EvalRequest {
            raw_input: raw_input.to_string(),
            parsed,
            action: EvalAction::Simplify,
            auto_store,
        }),
    }
}

fn parse_solve_input_as_equation_expr(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<ExprId, String> {
    let stmt = crate::parse_statement_or_session_ref(ctx, input)?;
    let parsed = match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => {
            let zero = ctx.num(0);
            ctx.call("Equal", vec![expr, zero])
        }
    };
    Ok(parsed)
}

/// Render eval-json input in LaTeX, formatting equations as `lhs = rhs`.
pub fn format_eval_input_latex(ctx: &cas_ast::Context, parsed: ExprId) -> String {
    if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
        let lhs_latex = LaTeXExpr {
            context: ctx,
            id: lhs,
        }
        .to_latex();
        let rhs_latex = LaTeXExpr {
            context: ctx,
            id: rhs,
        }
        .to_latex();
        format!("{} = {}", lhs_latex, rhs_latex)
    } else {
        LaTeXExpr {
            context: ctx,
            id: parsed,
        }
        .to_latex()
    }
}

/// Engine-level wrapper for building eval request from raw input.
pub fn build_eval_request_for_input_with_engine(
    engine: &mut crate::Engine,
    raw_input: &str,
    auto_store: bool,
) -> Result<EvalRequest, String> {
    build_eval_request_for_input(raw_input, &mut engine.simplifier.context, auto_store)
}

/// Engine-level wrapper for rendering eval input LaTeX.
pub fn format_eval_input_latex_with_engine(engine: &crate::Engine, parsed: ExprId) -> String {
    format_eval_input_latex(&engine.simplifier.context, parsed)
}

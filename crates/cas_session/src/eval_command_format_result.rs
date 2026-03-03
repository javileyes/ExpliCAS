use cas_ast::{Expr, ExprId};
use cas_formatter::root_style::ParseStyleSignals;

use crate::eval_command_types::{EvalCommandEvalView, EvalResultLine};

pub(crate) fn format_eval_result_line(
    context: &cas_ast::Context,
    parsed_expr: ExprId,
    result: &cas_solver::EvalResult,
    style_signals: &ParseStyleSignals,
) -> Option<EvalResultLine> {
    let style_prefs = cas_formatter::StylePreferences::from_expression_with_signals(
        context,
        parsed_expr,
        Some(style_signals),
    );

    match result {
        cas_solver::EvalResult::Expr(res) => {
            if let Expr::Function(name, args) = context.get(*res) {
                if context.is_builtin(*name, cas_ast::BuiltinFn::Equal) && args.len() == 2 {
                    return Some(EvalResultLine {
                        line: format!(
                            "Result: {} = {}",
                            cas_formatter::clean_display_string(&format!(
                                "{}",
                                cas_formatter::DisplayExprStyled::new(
                                    context,
                                    args[0],
                                    &style_prefs
                                )
                            )),
                            cas_formatter::clean_display_string(&format!(
                                "{}",
                                cas_formatter::DisplayExprStyled::new(
                                    context,
                                    args[1],
                                    &style_prefs
                                )
                            )),
                        ),
                        terminal: true,
                    });
                }
            }

            Some(EvalResultLine {
                line: format!(
                    "Result: {}",
                    cas_solver::display_expr_or_poly(context, *res)
                ),
                terminal: false,
            })
        }
        cas_solver::EvalResult::SolutionSet(solution_set) => Some(EvalResultLine {
            line: format!(
                "Result: {}",
                cas_solver::display_solution_set(context, solution_set)
            ),
            terminal: false,
        }),
        cas_solver::EvalResult::Set(_sols) => Some(EvalResultLine {
            line: "Result: Set(...)".to_string(),
            terminal: false,
        }),
        cas_solver::EvalResult::Bool(value) => Some(EvalResultLine {
            line: format!("Result: {}", value),
            terminal: false,
        }),
        cas_solver::EvalResult::None => None,
    }
}

pub(crate) fn format_eval_stored_entry_line(
    context: &cas_ast::Context,
    output: &EvalCommandEvalView,
) -> Option<String> {
    output.stored_id.map(|id| {
        format!(
            "#{id}: {}",
            cas_formatter::DisplayExpr {
                context,
                id: output.parsed
            }
        )
    })
}

pub(crate) fn format_eval_result_text(
    ctx: &cas_ast::Context,
    result: &cas_solver::EvalResult,
) -> String {
    match result {
        cas_solver::EvalResult::Expr(expr) => {
            if let Some(poly_str) = cas_solver::try_render_poly_result(ctx, *expr) {
                poly_str
            } else {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: *expr
                    }
                )
            }
        }
        cas_solver::EvalResult::Set(values) if !values.is_empty() => format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: values[0]
            }
        ),
        cas_solver::EvalResult::Bool(value) => value.to_string(),
        _ => "(no result)".to_string(),
    }
}

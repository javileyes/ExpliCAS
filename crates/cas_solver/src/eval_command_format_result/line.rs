use cas_ast::{Expr, ExprId};
use cas_formatter::root_style::ParseStyleSignals;

use crate::eval_command_types::EvalResultLine;

pub(super) fn format_eval_result_line(
    context: &cas_ast::Context,
    parsed_expr: ExprId,
    result: &crate::EvalResult,
    style_signals: &ParseStyleSignals,
) -> Option<EvalResultLine> {
    let style_prefs = cas_formatter::StylePreferences::from_expression_with_signals(
        context,
        parsed_expr,
        Some(style_signals),
    );

    match result {
        crate::EvalResult::Expr(res) => Some(format_expr_result_line(context, &style_prefs, *res)),
        crate::EvalResult::SolutionSet(solution_set) => Some(EvalResultLine {
            line: format!(
                "Result: {}",
                crate::display_solution_set(context, solution_set)
            ),
            terminal: false,
        }),
        crate::EvalResult::Set(_sols) => Some(EvalResultLine {
            line: "Result: Set(...)".to_string(),
            terminal: false,
        }),
        crate::EvalResult::Bool(value) => Some(EvalResultLine {
            line: format!("Result: {}", value),
            terminal: false,
        }),
        crate::EvalResult::None => None,
    }
}

fn format_expr_result_line(
    context: &cas_ast::Context,
    style_prefs: &cas_formatter::StylePreferences,
    result_expr: ExprId,
) -> EvalResultLine {
    if let Expr::Function(name, args) = context.get(result_expr) {
        if context.is_builtin(*name, cas_ast::BuiltinFn::Equal) && args.len() == 2 {
            return EvalResultLine {
                line: format!(
                    "Result: {} = {}",
                    render_styled_expr(context, args[0], style_prefs),
                    render_styled_expr(context, args[1], style_prefs),
                ),
                terminal: true,
            };
        }
    }

    EvalResultLine {
        line: format!(
            "Result: {}",
            crate::display_expr_or_poly(context, result_expr)
        ),
        terminal: false,
    }
}

fn render_styled_expr(
    context: &cas_ast::Context,
    expr: ExprId,
    style_prefs: &cas_formatter::StylePreferences,
) -> String {
    cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExprStyled::new(context, expr, style_prefs)
    ))
}

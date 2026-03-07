use cas_ast::{Expr, ExprId};

use crate::eval_command_types::EvalResultLine;

pub(super) fn format_expr_result_line(
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

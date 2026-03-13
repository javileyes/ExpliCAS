mod expr;
mod scalar;

use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;

use crate::command_api::eval::EvalResultLine;

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
        crate::EvalResult::Expr(res) => {
            Some(expr::format_expr_result_line(context, &style_prefs, *res))
        }
        crate::EvalResult::SolutionSet(solution_set) => Some(
            scalar::format_solution_set_result_line(context, solution_set),
        ),
        crate::EvalResult::Set(_sols) => Some(scalar::format_set_result_line()),
        crate::EvalResult::Bool(value) => Some(scalar::format_bool_result_line(*value)),
        crate::EvalResult::None => None,
    }
}

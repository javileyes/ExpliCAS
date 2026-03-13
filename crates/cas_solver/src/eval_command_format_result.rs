mod line;
mod stored;
mod text;

use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;

use crate::command_api::eval::{EvalCommandEvalView, EvalResultLine};

pub(crate) fn format_eval_result_line(
    context: &cas_ast::Context,
    parsed_expr: ExprId,
    result: &crate::EvalResult,
    style_signals: &ParseStyleSignals,
) -> Option<EvalResultLine> {
    self::line::format_eval_result_line(context, parsed_expr, result, style_signals)
}

pub(crate) fn format_eval_stored_entry_line(
    context: &cas_ast::Context,
    output: &EvalCommandEvalView,
) -> Option<String> {
    self::stored::format_eval_stored_entry_line(context, output)
}

pub(crate) fn format_eval_result_text(
    ctx: &cas_ast::Context,
    result: &crate::EvalResult,
) -> String {
    self::text::format_eval_result_text(ctx, result)
}

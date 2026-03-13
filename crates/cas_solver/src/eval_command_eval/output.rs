use cas_formatter::root_style::ParseStyleSignals;

use crate::command_api::eval::{EvalCommandEvalView, EvalCommandOutput};
use crate::eval_command_format_metadata::format_eval_metadata_lines;
use crate::eval_command_format_result::{format_eval_result_line, format_eval_stored_entry_line};

pub(super) fn build_eval_command_output(
    context: &mut cas_ast::Context,
    eval_options: crate::EvalOptions,
    eval_view: EvalCommandEvalView,
    style_signals: ParseStyleSignals,
    debug_mode: bool,
) -> EvalCommandOutput {
    let metadata = format_eval_metadata_lines(
        context,
        &eval_view,
        eval_options.requires_display,
        debug_mode,
        eval_options.hints_enabled,
        eval_options.shared.semantics.domain_mode,
        eval_options.shared.assumption_reporting,
    );
    let stored_entry_line = format_eval_stored_entry_line(context, &eval_view);
    let result_line =
        format_eval_result_line(context, eval_view.parsed, &eval_view.result, &style_signals);

    EvalCommandOutput {
        resolved_expr: eval_view.resolved,
        style_signals,
        steps: eval_view.steps,
        stored_entry_line,
        metadata,
        result_line,
    }
}

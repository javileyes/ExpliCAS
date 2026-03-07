use crate::eval_command_types::EvalCommandEvalView;

pub(super) fn format_eval_stored_entry_line(
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

use crate::{
    format_history_entry_inspection_lines, HistoryEntryDetails, HistoryEntryInspection,
    HistoryExprInspection,
};

/// Format `show` command lines from a pre-computed inspection.
pub fn format_show_history_command_lines<F, M>(
    inspection: &HistoryEntryInspection,
    render_expr: F,
    mut metadata_lines: M,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
    M: FnMut(&HistoryExprInspection) -> Vec<String>,
{
    let mut lines = format_history_entry_inspection_lines(inspection, render_expr);
    if let HistoryEntryDetails::Expr(expr_info) = &inspection.details {
        lines.extend(metadata_lines(expr_info));
    }
    lines
}

/// Format `show` command lines from a pre-computed inspection using explicit context.
pub fn format_show_history_command_lines_with_context<M>(
    inspection: &HistoryEntryInspection,
    context: &cas_ast::Context,
    mut metadata_lines: M,
) -> Vec<String>
where
    M: FnMut(&cas_ast::Context, &HistoryExprInspection) -> Vec<String>,
{
    let mut lines = format_history_entry_inspection_lines(inspection, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    });
    if let HistoryEntryDetails::Expr(expr_info) = &inspection.details {
        lines.extend(metadata_lines(context, expr_info));
    }
    lines
}

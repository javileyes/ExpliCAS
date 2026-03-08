use cas_ast::ExprId;

use crate::{HistoryEntryDetails, HistoryEntryInspection};

/// Format history-entry inspection lines using an expression renderer callback.
pub fn format_history_entry_inspection_lines<F>(
    inspection: &HistoryEntryInspection,
    mut render_expr: F,
) -> Vec<String>
where
    F: FnMut(ExprId) -> String,
{
    let mut lines = vec![
        format!("Entry #{}:", inspection.id),
        format!("  Type:       {}", inspection.type_str),
        format!("  Raw:        {}", inspection.raw_text),
    ];

    match &inspection.details {
        HistoryEntryDetails::Expr(expr_info) => {
            lines.push(format!("  Parsed:     {}", render_expr(expr_info.parsed)));
            if let Some(resolved) = expr_info.resolved {
                lines.push(format!("  Resolved:   {}", render_expr(resolved)));
            }
            if let Some(simplified) = expr_info.simplified {
                lines.push(format!("  Simplified: {}", render_expr(simplified)));
            }
        }
        HistoryEntryDetails::Eq { lhs, rhs } => {
            lines.push(format!("  LHS:        {}", render_expr(*lhs)));
            lines.push(format!("  RHS:        {}", render_expr(*rhs)));
            lines.push(String::new());
            lines.push("  Note: When used as expression, this becomes (LHS - RHS).".to_string());
        }
    }

    lines
}

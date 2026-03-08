#[cfg(test)]
mod tests {
    use crate::{evaluate_history_command_lines, HistoryOverviewEntry, HistoryOverviewKind};

    #[test]
    fn evaluate_history_command_lines_empty_message() {
        let lines = evaluate_history_command_lines(&[], |_id| "<expr>".to_string());
        assert_eq!(lines, vec!["No entries in session history.".to_string()]);
    }

    #[test]
    fn evaluate_history_command_lines_renders_entries() {
        let entries = vec![HistoryOverviewEntry {
            id: 1,
            kind: HistoryOverviewKind::Expr {
                expr: cas_ast::ExprId::from_raw(10),
            },
        }];
        let lines = evaluate_history_command_lines(&entries, |id| format!("E{}", id.index()));
        assert_eq!(lines[0], "Session history (1 entries):");
        assert_eq!(lines[1], "  #1   [Expr] E10");
    }
}

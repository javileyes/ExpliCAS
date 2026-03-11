#[cfg(test)]
mod tests {
    use crate::{EntryKind, SessionState};
    #[allow(unused_imports)]
    use cas_solver::session_api::{
        formatting::*, options::*, runtime::*, session_support::*, symbolic_commands::*, types::*,
    };

    #[test]
    fn parse_history_ids_accepts_hash_prefix() {
        let ids = parse_history_ids("#1 #2 3 nope");
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn delete_history_entries_errors_without_ids() {
        let mut state = SessionState::new();
        let err = delete_history_entries(&mut state, "abc").expect_err("expected no ids error");
        assert_eq!(err, DeleteHistoryError::NoValidIds);
    }

    #[test]
    fn delete_history_entries_removes_existing_entries() {
        let mut state = SessionState::new();
        let mut ctx = cas_ast::Context::new();
        let x = cas_parser::parse("x", &mut ctx).expect("parse x");
        let y = cas_parser::parse("y", &mut ctx).expect("parse y");
        state.history_push(EntryKind::Expr(x), "x");
        state.history_push(EntryKind::Expr(y), "y");

        let result = delete_history_entries(&mut state, "#1 #99").expect("delete");
        assert_eq!(result.requested_ids, vec![1, 99]);
        assert_eq!(result.removed_count, 1);
        assert_eq!(state.history_len(), 1);
    }

    #[test]
    fn history_overview_entries_returns_typed_view() {
        let mut state = SessionState::new();
        let mut ctx = cas_ast::Context::new();
        let x = cas_parser::parse("x", &mut ctx).expect("parse x");
        let y = cas_parser::parse("y", &mut ctx).expect("parse y");
        state.history_push(EntryKind::Expr(x), "x");
        state.history_push(EntryKind::Eq { lhs: x, rhs: y }, "x = y");

        let overview = history_overview_entries(&state);
        assert_eq!(overview.len(), 2);
        assert!(matches!(overview[0].kind, HistoryOverviewKind::Expr { expr } if expr == x));
        assert!(matches!(
            overview[1].kind,
            HistoryOverviewKind::Eq { lhs, rhs } if lhs == x && rhs == y
        ));
    }

    #[test]
    fn format_delete_history_result_message_lists_ids() {
        let message = format_delete_history_result_message(&DeleteHistoryResult {
            requested_ids: vec![1, 3],
            removed_count: 1,
        });
        assert_eq!(message, "Deleted 1 entry/entries: #1, #3");
    }

    #[test]
    fn format_delete_history_error_message_no_valid_ids() {
        let message = format_delete_history_error_message(&DeleteHistoryError::NoValidIds);
        assert!(message.contains("Use 'del #1 #2'"));
    }

    #[test]
    fn history_empty_message_is_stable() {
        assert_eq!(history_empty_message(), "No entries in session history.");
    }

    #[test]
    fn format_history_overview_lines_renders_entries() {
        let entries = vec![
            HistoryOverviewEntry {
                id: 1,
                kind: HistoryOverviewKind::Expr {
                    expr: cas_ast::ExprId::from_raw(10),
                },
            },
            HistoryOverviewEntry {
                id: 2,
                kind: HistoryOverviewKind::Eq {
                    lhs: cas_ast::ExprId::from_raw(11),
                    rhs: cas_ast::ExprId::from_raw(12),
                },
            },
        ];
        let lines = format_history_overview_lines(&entries, |id| format!("E{}", id.index()));
        assert_eq!(lines[0], "Session history (2 entries):");
        assert_eq!(lines[1], "  #1   [Expr] E10");
        assert_eq!(lines[2], "  #2   [Eq  ] E11 = E12");
    }
}

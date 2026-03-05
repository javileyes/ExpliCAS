#[cfg(test)]
mod tests {
    use crate::{
        evaluate_history_command_lines_from_history,
        evaluate_history_command_lines_from_history_with_context, HistoryEntryKindRaw,
        HistoryEntryRaw, HistoryOverviewContext,
    };

    struct TestHistoryContext;

    impl HistoryOverviewContext for TestHistoryContext {
        fn history_entries_raw(&self) -> Vec<HistoryEntryRaw> {
            vec![HistoryEntryRaw {
                id: 1,
                kind: HistoryEntryKindRaw::Expr(cas_ast::ExprId::from_raw(10)),
            }]
        }
    }

    #[test]
    fn evaluate_history_command_lines_from_history_renders_entries() {
        let lines = evaluate_history_command_lines_from_history(&TestHistoryContext, |id| {
            format!("E{}", id.index())
        });
        assert_eq!(lines[0], "Session history (1 entries):");
        assert_eq!(lines[1], "  #1   [Expr] E10");
    }

    #[test]
    fn evaluate_history_command_lines_from_history_with_context_renders_entries() {
        let mut ast_context = cas_ast::Context::new();
        let x = ast_context.var("x");
        struct Ctx(cas_ast::ExprId);
        impl HistoryOverviewContext for Ctx {
            fn history_entries_raw(&self) -> Vec<HistoryEntryRaw> {
                vec![HistoryEntryRaw {
                    id: 1,
                    kind: HistoryEntryKindRaw::Expr(self.0),
                }]
            }
        }
        let lines = evaluate_history_command_lines_from_history_with_context(&Ctx(x), &ast_context);
        assert_eq!(lines[0], "Session history (1 entries):");
        assert_eq!(lines[1], "  #1   [Expr] x");
    }
}

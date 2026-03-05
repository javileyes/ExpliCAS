#[cfg(test)]
mod tests {
    use crate::{
        history_overview_entries, HistoryEntryKindRaw, HistoryEntryRaw, HistoryOverviewContext,
        HistoryOverviewKind,
    };

    struct TestHistoryOverviewContext;

    impl HistoryOverviewContext for TestHistoryOverviewContext {
        fn history_entries_raw(&self) -> Vec<HistoryEntryRaw> {
            vec![
                HistoryEntryRaw {
                    id: 1,
                    kind: HistoryEntryKindRaw::Expr(cas_ast::ExprId::from_raw(10)),
                },
                HistoryEntryRaw {
                    id: 2,
                    kind: HistoryEntryKindRaw::Eq {
                        lhs: cas_ast::ExprId::from_raw(11),
                        rhs: cas_ast::ExprId::from_raw(12),
                    },
                },
            ]
        }
    }

    #[test]
    fn history_overview_entries_maps_raw_entries() {
        let context = TestHistoryOverviewContext;
        let overview = history_overview_entries(&context);
        assert_eq!(overview.len(), 2);
        assert!(matches!(
            overview[0].kind,
            HistoryOverviewKind::Expr { expr } if expr == cas_ast::ExprId::from_raw(10)
        ));
        assert!(matches!(
            overview[1].kind,
            HistoryOverviewKind::Eq { lhs, rhs }
            if lhs == cas_ast::ExprId::from_raw(11) && rhs == cas_ast::ExprId::from_raw(12)
        ));
    }
}

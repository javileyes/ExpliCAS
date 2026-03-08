#[cfg(test)]
mod tests {
    use crate::{
        delete_history_entries, evaluate_delete_history_command_message, DeleteHistoryError,
        HistoryDeleteContext,
    };

    #[derive(Debug, Default)]
    struct TestHistoryContext {
        ids: Vec<u64>,
    }

    impl HistoryDeleteContext for TestHistoryContext {
        fn history_len(&self) -> usize {
            self.ids.len()
        }

        fn history_remove(&mut self, ids: &[u64]) {
            self.ids.retain(|id| !ids.contains(id));
        }
    }

    #[test]
    fn delete_history_entries_rejects_missing_ids() {
        let mut context = TestHistoryContext { ids: vec![1, 2, 3] };
        let err = delete_history_entries(&mut context, "del nope").expect_err("expected error");
        assert_eq!(err, DeleteHistoryError::NoValidIds);
    }

    #[test]
    fn delete_history_entries_reports_removed_count() {
        let mut context = TestHistoryContext { ids: vec![1, 2, 3] };
        let result = delete_history_entries(&mut context, "del 1 3").expect("delete");
        assert_eq!(result.requested_ids, vec![1, 3]);
        assert_eq!(result.removed_count, 2);
        assert_eq!(context.ids, vec![2]);
    }

    #[test]
    fn evaluate_delete_history_command_message_formats_error() {
        let mut context = TestHistoryContext { ids: vec![1, 2, 3] };
        let message = evaluate_delete_history_command_message(&mut context, "del nope");
        assert!(message.contains("No valid IDs specified"));
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        binding_overview_entries, clear_bindings_command, BindingsContext, ClearBindingsResult,
    };

    #[derive(Default)]
    struct TestBindingsContext {
        entries: Vec<(String, cas_ast::ExprId)>,
    }

    impl BindingsContext for TestBindingsContext {
        fn binding_count(&self) -> usize {
            self.entries.len()
        }

        fn clear_bindings(&mut self) {
            self.entries.clear();
        }

        fn unset_binding(&mut self, name: &str) -> bool {
            let before = self.entries.len();
            self.entries.retain(|(n, _)| n != name);
            before != self.entries.len()
        }

        fn bindings(&self) -> Vec<(String, cas_ast::ExprId)> {
            self.entries.clone()
        }
    }

    #[test]
    fn clear_bindings_command_all_mode() {
        let mut context = TestBindingsContext {
            entries: vec![
                ("a".to_string(), cas_ast::ExprId::from_raw(10)),
                ("b".to_string(), cas_ast::ExprId::from_raw(11)),
            ],
        };

        let result = clear_bindings_command(&mut context, "clear");
        assert_eq!(result, ClearBindingsResult::All { cleared_count: 2 });
        assert_eq!(context.binding_count(), 0);
    }

    #[test]
    fn clear_bindings_command_selected_mode_tracks_missing() {
        let mut context = TestBindingsContext {
            entries: vec![("a".to_string(), cas_ast::ExprId::from_raw(10))],
        };

        let result = clear_bindings_command(&mut context, "clear a z");
        assert_eq!(
            result,
            ClearBindingsResult::Selected {
                cleared_count: 1,
                missing_names: vec!["z".to_string()],
            }
        );
        assert_eq!(context.binding_count(), 0);
    }

    #[test]
    fn binding_overview_entries_returns_sorted_bindings() {
        let context = TestBindingsContext {
            entries: vec![
                ("a".to_string(), cas_ast::ExprId::from_raw(10)),
                ("b".to_string(), cas_ast::ExprId::from_raw(11)),
            ],
        };

        let entries = binding_overview_entries(&context);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "a");
        assert_eq!(entries[1].name, "b");
    }
}

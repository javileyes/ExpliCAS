#[cfg(test)]
mod tests {
    use crate::bindings_command::BindingsContext;
    use crate::session_api::bindings::{evaluate_clear_command_lines, evaluate_vars_command_lines};

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
    fn evaluate_clear_bindings_command_lines_all_mode() {
        let mut context = TestBindingsContext {
            entries: vec![("a".to_string(), cas_ast::ExprId::from_raw(10))],
        };
        let lines = evaluate_clear_command_lines(&mut context, "clear");
        assert_eq!(lines, vec!["Cleared 1 variable(s).".to_string()]);
    }

    #[test]
    fn evaluate_vars_command_lines_from_bindings_renders_lines() {
        let context = TestBindingsContext {
            entries: vec![("a".to_string(), cas_ast::ExprId::from_raw(10))],
        };
        let lines = evaluate_vars_command_lines(&context, |expr| format!("E{}", expr.index()));
        assert_eq!(lines[0], "Variables:");
        assert_eq!(lines[1], "  a = E10");
    }
}

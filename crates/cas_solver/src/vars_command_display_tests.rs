#[cfg(test)]
mod tests {
    use crate::{evaluate_vars_command_lines, BindingOverviewEntry};

    #[test]
    fn evaluate_vars_command_lines_empty_message() {
        let lines = evaluate_vars_command_lines(&[], |_id| "<expr>".to_string());
        assert_eq!(lines, vec!["No variables defined.".to_string()]);
    }

    #[test]
    fn evaluate_vars_command_lines_renders_entries() {
        let entries = vec![BindingOverviewEntry {
            name: "a".to_string(),
            expr: cas_ast::ExprId::from_raw(10),
        }];
        let lines = evaluate_vars_command_lines(&entries, |id| format!("E{}", id.index()));
        assert_eq!(lines[0], "Variables:");
        assert_eq!(lines[1], "  a = E10");
    }
}

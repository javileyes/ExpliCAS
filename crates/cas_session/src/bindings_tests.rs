#[cfg(test)]
mod tests {
    use crate::SessionState;
    #[allow(unused_imports)]
    use cas_solver::session_api::{
        formatting::*, options::*, runtime::*, session_support::*, symbolic_commands::*, types::*,
    };

    #[test]
    fn clear_bindings_command_all_mode() {
        let mut state = SessionState::new();
        let mut ctx = cas_ast::Context::new();
        let x = cas_parser::parse("x + 1", &mut ctx).expect("parse");
        state.set_binding("a", x);
        state.set_binding("b", x);

        let result = clear_bindings_command(&mut state, "clear");
        assert_eq!(result, ClearBindingsResult::All { cleared_count: 2 });
        assert_eq!(state.binding_count(), 0);
    }

    #[test]
    fn clear_bindings_command_selected_mode_tracks_missing() {
        let mut state = SessionState::new();
        let mut ctx = cas_ast::Context::new();
        let x = cas_parser::parse("x + 1", &mut ctx).expect("parse");
        state.set_binding("a", x);

        let result = clear_bindings_command(&mut state, "clear a z");
        assert_eq!(
            result,
            ClearBindingsResult::Selected {
                cleared_count: 1,
                missing_names: vec!["z".to_string()],
            }
        );
        assert_eq!(state.binding_count(), 0);
    }

    #[test]
    fn binding_overview_entries_returns_sorted_bindings() {
        let mut state = SessionState::new();
        let mut ctx = cas_ast::Context::new();
        let x = cas_parser::parse("x", &mut ctx).expect("parse");
        let y = cas_parser::parse("y", &mut ctx).expect("parse");
        state.set_binding("b", y);
        state.set_binding("a", x);

        let entries = binding_overview_entries(&state);
        assert_eq!(
            entries,
            vec![
                BindingOverviewEntry {
                    name: "a".to_string(),
                    expr: x,
                },
                BindingOverviewEntry {
                    name: "b".to_string(),
                    expr: y,
                },
            ]
        );
    }

    #[test]
    fn format_clear_bindings_result_lines_all_empty() {
        let lines =
            format_clear_bindings_result_lines(&ClearBindingsResult::All { cleared_count: 0 });
        assert_eq!(lines, vec!["No variables to clear.".to_string()]);
    }

    #[test]
    fn format_clear_bindings_result_lines_selected_with_warning() {
        let lines = format_clear_bindings_result_lines(&ClearBindingsResult::Selected {
            cleared_count: 1,
            missing_names: vec!["z".to_string()],
        });
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "Warning: 'z' was not defined");
        assert_eq!(lines[1], "Cleared 1 variable(s).");
    }

    #[test]
    fn vars_empty_message_is_stable() {
        assert_eq!(vars_empty_message(), "No variables defined.");
    }

    #[test]
    fn format_binding_overview_lines_renders_entries() {
        let entries = vec![
            BindingOverviewEntry {
                name: "a".to_string(),
                expr: cas_ast::ExprId::from_raw(10),
            },
            BindingOverviewEntry {
                name: "b".to_string(),
                expr: cas_ast::ExprId::from_raw(11),
            },
        ];
        let lines = format_binding_overview_lines(&entries, |id| format!("E{}", id.index()));
        assert_eq!(lines[0], "Variables:");
        assert_eq!(lines[1], "  a = E10");
        assert_eq!(lines[2], "  b = E11");
    }
}

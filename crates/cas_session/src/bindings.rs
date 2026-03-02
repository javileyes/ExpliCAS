use crate::SessionState;

/// Lightweight binding view for presentation layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingOverviewEntry {
    pub name: String,
    pub expr: cas_ast::ExprId,
}

/// Result of applying a `clear`-style binding command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClearBindingsResult {
    All {
        cleared_count: usize,
    },
    Selected {
        cleared_count: usize,
        missing_names: Vec<String>,
    },
}

/// Message used when no variables exist.
pub fn vars_empty_message() -> &'static str {
    "No variables defined."
}

/// Format binding overview lines using an expression renderer callback.
pub fn format_binding_overview_lines<F>(
    entries: &[BindingOverviewEntry],
    mut render_expr: F,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let mut lines = vec!["Variables:".to_string()];
    for entry in entries {
        lines.push(format!("  {} = {}", entry.name, render_expr(entry.expr)));
    }
    lines
}

/// Format clear-bindings command result into output lines.
pub fn format_clear_bindings_result_lines(result: &ClearBindingsResult) -> Vec<String> {
    match result {
        ClearBindingsResult::All { cleared_count } => {
            if *cleared_count == 0 {
                vec!["No variables to clear.".to_string()]
            } else {
                vec![format!("Cleared {} variable(s).", cleared_count)]
            }
        }
        ClearBindingsResult::Selected {
            cleared_count,
            missing_names,
        } => {
            let mut lines: Vec<String> = missing_names
                .iter()
                .map(|name| format!("Warning: '{}' was not defined", name))
                .collect();
            if *cleared_count > 0 {
                lines.push(format!("Cleared {} variable(s).", cleared_count));
            }
            lines
        }
    }
}

/// Apply a `clear` command over session bindings.
///
/// Accepts either:
/// - `clear` (clear all bindings)
/// - `clear <name> [<name> ...]` (clear selected bindings)
pub fn clear_bindings_command(state: &mut SessionState, input: &str) -> ClearBindingsResult {
    let trimmed = input.trim();
    let all_mode = trimmed == "clear" || trimmed.is_empty();

    if all_mode {
        let cleared_count = state.binding_count();
        state.clear_bindings();
        return ClearBindingsResult::All { cleared_count };
    }

    let names_part = trimmed.strip_prefix("clear").unwrap_or(trimmed).trim();
    let names: Vec<&str> = names_part.split_whitespace().collect();

    let mut cleared_count = 0;
    let mut missing_names = Vec::new();
    for name in names {
        if state.unset_binding(name) {
            cleared_count += 1;
        } else {
            missing_names.push(name.to_string());
        }
    }

    ClearBindingsResult::Selected {
        cleared_count,
        missing_names,
    }
}

/// Return a stable, presentation-friendly view of bindings.
pub fn binding_overview_entries(state: &SessionState) -> Vec<BindingOverviewEntry> {
    state
        .bindings()
        .into_iter()
        .map(|(name, expr)| BindingOverviewEntry { name, expr })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        binding_overview_entries, clear_bindings_command, format_binding_overview_lines,
        format_clear_bindings_result_lines, vars_empty_message, BindingOverviewEntry,
        ClearBindingsResult,
    };
    use crate::SessionState;

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

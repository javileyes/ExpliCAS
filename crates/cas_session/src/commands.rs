use crate::{
    apply_assignment, clear_bindings_command, delete_history_entries,
    format_assignment_error_message, format_assignment_success_message,
    format_binding_overview_lines, format_clear_bindings_result_lines,
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_entry_inspection_lines, format_history_overview_lines,
    format_inspect_history_entry_error_message, format_let_assignment_parse_error_message,
    history_empty_message, history_overview_entries, inspect_history_entry_input,
    parse_let_assignment_input, vars_empty_message, HistoryEntryDetails, HistoryEntryInspection,
    HistoryExprInspection, SessionState,
};

/// Evaluate `vars` command lines using an expression renderer callback.
pub fn evaluate_vars_command_lines<F>(state: &SessionState, render_expr: F) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let bindings = crate::binding_overview_entries(state);
    if bindings.is_empty() {
        vec![vars_empty_message().to_string()]
    } else {
        format_binding_overview_lines(&bindings, render_expr)
    }
}

/// Evaluate `history` command lines using an expression renderer callback.
pub fn evaluate_history_command_lines<F>(state: &SessionState, render_expr: F) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
{
    let entries = history_overview_entries(state);
    if entries.is_empty() {
        vec![history_empty_message().to_string()]
    } else {
        format_history_overview_lines(&entries, render_expr)
    }
}

/// Evaluate `clear` command and return output lines.
pub fn evaluate_clear_command_lines(state: &mut SessionState, input: &str) -> Vec<String> {
    let result = clear_bindings_command(state, input);
    format_clear_bindings_result_lines(&result)
}

/// Evaluate `del` command and return a user-facing message.
pub fn evaluate_delete_history_command_message(state: &mut SessionState, input: &str) -> String {
    match delete_history_entries(state, input) {
        Ok(result) => format_delete_history_result_message(&result),
        Err(error) => format_delete_history_error_message(&error),
    }
}

/// Evaluate `show` command and return output lines.
///
/// `render_expr` renders expressions in the same style as the caller context.
/// `metadata_lines` can append section lines for expression entries.
pub fn inspect_show_history_command(
    state: &mut SessionState,
    engine: &mut cas_engine::Engine,
    input: &str,
) -> Result<HistoryEntryInspection, String> {
    inspect_history_entry_input(state, engine, input)
        .map_err(|error| format_inspect_history_entry_error_message(&error))
}

/// Format `show` command lines from a pre-computed inspection.
pub fn format_show_history_command_lines<F, M>(
    inspection: &HistoryEntryInspection,
    render_expr: F,
    mut metadata_lines: M,
) -> Vec<String>
where
    F: FnMut(cas_ast::ExprId) -> String,
    M: FnMut(&HistoryExprInspection) -> Vec<String>,
{
    let mut lines = format_history_entry_inspection_lines(inspection, render_expr);
    if let HistoryEntryDetails::Expr(expr_info) = &inspection.details {
        lines.extend(metadata_lines(expr_info));
    }
    lines
}

/// Evaluate `show` command and return output lines.
pub fn evaluate_show_history_command_lines<F, M>(
    state: &mut SessionState,
    engine: &mut cas_engine::Engine,
    input: &str,
    render_expr: F,
    metadata_lines: M,
) -> Result<Vec<String>, String>
where
    F: FnMut(cas_ast::ExprId) -> String,
    M: FnMut(&HistoryExprInspection) -> Vec<String>,
{
    let inspection = inspect_show_history_command(state, engine, input)?;
    Ok(format_show_history_command_lines(
        &inspection,
        render_expr,
        metadata_lines,
    ))
}

/// Successful output payload for assignment-style commands (`let`, `:=`, direct assign).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentCommandOutput {
    pub name: String,
    pub expr: cas_ast::ExprId,
    pub lazy: bool,
}

/// Evaluate assignment command pieces and return a typed output payload.
pub fn evaluate_assignment_command(
    state: &mut SessionState,
    simplifier: &mut cas_engine::Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<AssignmentCommandOutput, String> {
    match apply_assignment(state, simplifier, name, expr_str, lazy) {
        Ok(expr) => Ok(AssignmentCommandOutput {
            name: name.to_string(),
            expr,
            lazy,
        }),
        Err(error) => Err(format_assignment_error_message(&error)),
    }
}

/// Evaluate `let ...` command tail and return assignment output payload.
pub fn evaluate_let_assignment_command(
    state: &mut SessionState,
    simplifier: &mut cas_engine::Simplifier,
    input: &str,
) -> Result<AssignmentCommandOutput, String> {
    let parsed = parse_let_assignment_input(input)
        .map_err(|error| format_let_assignment_parse_error_message(&error))?;
    evaluate_assignment_command(state, simplifier, parsed.name, parsed.expr, parsed.lazy)
}

/// Format assignment output payload once caller rendered the expression.
pub fn format_assignment_command_output_message(
    output: &AssignmentCommandOutput,
    rendered_expr: &str,
) -> String {
    format_assignment_success_message(&output.name, rendered_expr, output.lazy)
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_assignment_command, evaluate_clear_command_lines,
        evaluate_delete_history_command_message, evaluate_history_command_lines,
        evaluate_let_assignment_command, evaluate_show_history_command_lines,
        evaluate_vars_command_lines, format_assignment_command_output_message,
        format_show_history_command_lines, inspect_show_history_command,
    };
    use crate::{EntryKind, SessionState};

    #[test]
    fn evaluate_vars_command_lines_empty() {
        let state = SessionState::new();
        let lines = evaluate_vars_command_lines(&state, |_id| "<expr>".to_string());
        assert_eq!(lines, vec!["No variables defined.".to_string()]);
    }

    #[test]
    fn evaluate_history_command_lines_empty() {
        let state = SessionState::new();
        let lines = evaluate_history_command_lines(&state, |_id| "<expr>".to_string());
        assert_eq!(lines, vec!["No entries in session history.".to_string()]);
    }

    #[test]
    fn evaluate_clear_command_lines_returns_summary() {
        let mut state = SessionState::new();
        let lines = evaluate_clear_command_lines(&mut state, "clear");
        assert_eq!(lines, vec!["No variables to clear.".to_string()]);
    }

    #[test]
    fn evaluate_delete_history_command_message_for_invalid_ids() {
        let mut state = SessionState::new();
        let msg = evaluate_delete_history_command_message(&mut state, "del nope");
        assert!(msg.contains("No valid IDs"));
    }

    #[test]
    fn evaluate_show_history_command_lines_appends_metadata() {
        let mut state = SessionState::new();
        let mut engine = cas_engine::Engine::new();
        let expr = cas_parser::parse("x + x", &mut engine.simplifier.context).expect("parse expr");
        let id = state.history_push(EntryKind::Expr(expr), "x + x".to_string());

        let lines = evaluate_show_history_command_lines(
            &mut state,
            &mut engine,
            &format!("#{}", id),
            |_id| "expr".to_string(),
            |_expr_info| vec!["meta".to_string()],
        )
        .expect("show lines");

        assert!(lines.iter().any(|line| line == "meta"));
    }

    #[test]
    fn inspect_show_history_command_reports_invalid_id() {
        let mut state = SessionState::new();
        let mut engine = cas_engine::Engine::new();
        let err = inspect_show_history_command(&mut state, &mut engine, "bad").expect_err("error");
        assert!(err.contains("Invalid entry ID"));
    }

    #[test]
    fn format_show_history_command_lines_appends_metadata() {
        let inspection = crate::HistoryEntryInspection {
            id: 1,
            type_str: "Expression".to_string(),
            raw_text: "x+x".to_string(),
            details: crate::HistoryEntryDetails::Expr(crate::HistoryExprInspection {
                parsed: cas_ast::ExprId::from_raw(1),
                resolved: None,
                simplified: None,
                required_conditions: Vec::new(),
                domain_warnings: Vec::new(),
                blocked_hints: Vec::new(),
            }),
        };
        let lines = format_show_history_command_lines(
            &inspection,
            |_id| "expr".to_string(),
            |_expr_info| vec!["meta".to_string()],
        );
        assert!(lines.iter().any(|line| line == "meta"));
    }

    #[test]
    fn evaluate_assignment_command_success() {
        let mut state = SessionState::new();
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        let out = evaluate_assignment_command(&mut state, &mut simplifier, "a", "x + x", true)
            .expect("assign");

        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: out.expr
            }
        );
        let message = format_assignment_command_output_message(&out, &rendered);
        assert!(message.starts_with("a "));
    }

    #[test]
    fn evaluate_let_assignment_command_parse_error() {
        let mut state = SessionState::new();
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        let err = evaluate_let_assignment_command(&mut state, &mut simplifier, "x + y")
            .expect_err("let parse error");
        assert!(err.contains("Usage:"));
    }
}

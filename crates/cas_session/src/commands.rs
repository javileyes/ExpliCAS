use crate::{
    apply_assignment, clear_bindings_command, delete_history_entries,
    format_assignment_error_message, format_assignment_success_message,
    format_binding_overview_lines, format_clear_bindings_result_lines,
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_entry_inspection_lines, format_history_overview_lines,
    format_let_assignment_parse_error_message, history_empty_message, history_overview_entries,
    parse_let_assignment_input, vars_empty_message, HistoryEntryDetails, HistoryEntryInspection,
    HistoryExprInspection, SessionState,
};

/// Result of applying a `cache` command against engine profile cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCacheCommandResult {
    Status { cached_profiles: usize },
    Cleared,
    Unknown { command: String },
}

enum ProfileCacheCommandInput {
    Status,
    Clear,
    Unknown(String),
}

fn parse_profile_cache_command_input(line: &str) -> ProfileCacheCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1).copied() {
        None | Some("status") => ProfileCacheCommandInput::Status,
        Some("clear") => ProfileCacheCommandInput::Clear,
        Some(other) => ProfileCacheCommandInput::Unknown(other.to_string()),
    }
}

/// Apply a `cache` command line to an engine profile cache.
pub fn apply_profile_cache_command(
    engine: &mut cas_engine::Engine,
    line: &str,
) -> ProfileCacheCommandResult {
    match parse_profile_cache_command_input(line) {
        ProfileCacheCommandInput::Status => ProfileCacheCommandResult::Status {
            cached_profiles: engine.profile_cache_len(),
        },
        ProfileCacheCommandInput::Clear => {
            engine.clear_profile_cache();
            ProfileCacheCommandResult::Cleared
        }
        ProfileCacheCommandInput::Unknown(command) => {
            ProfileCacheCommandResult::Unknown { command }
        }
    }
}

/// Render a `cache` command result into output lines for UI/frontends.
pub fn format_profile_cache_command_lines(result: &ProfileCacheCommandResult) -> Vec<String> {
    match result {
        ProfileCacheCommandResult::Status {
            cached_profiles: count,
        } => {
            let mut lines = vec![format!("Profile Cache: {} profiles cached", count)];
            if *count == 0 {
                lines.push("  (empty - profiles will be built on first eval)".to_string());
            } else {
                lines.push("  (profiles are reused across evaluations)".to_string());
            }
            vec![lines.join("\n")]
        }
        ProfileCacheCommandResult::Cleared => vec!["Profile cache cleared.".to_string()],
        ProfileCacheCommandResult::Unknown { command } => vec![
            format!("Unknown cache command: {}", command),
            "Usage: cache [status|clear]".to_string(),
        ],
    }
}

/// Parsed input for the `profile` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCommandInput {
    ShowReport,
    Enable,
    Disable,
    Clear,
    Invalid,
}

/// Normalized result for `profile` command handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCommandResult {
    ShowReport,
    SetEnabled { enabled: bool, message: String },
    Clear { message: String },
    Invalid { message: String },
}

/// Parse raw `profile ...` command input.
pub fn parse_profile_command_input(line: &str) -> ProfileCommandInput {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() == 1 {
        return ProfileCommandInput::ShowReport;
    }
    match parts[1] {
        "enable" => ProfileCommandInput::Enable,
        "disable" => ProfileCommandInput::Disable,
        "clear" => ProfileCommandInput::Clear,
        _ => ProfileCommandInput::Invalid,
    }
}

/// Evaluate a `profile` command into effect + message.
pub fn evaluate_profile_command_input(line: &str) -> ProfileCommandResult {
    match parse_profile_command_input(line) {
        ProfileCommandInput::ShowReport => ProfileCommandResult::ShowReport,
        ProfileCommandInput::Enable => ProfileCommandResult::SetEnabled {
            enabled: true,
            message: "Profiler enabled.".to_string(),
        },
        ProfileCommandInput::Disable => ProfileCommandResult::SetEnabled {
            enabled: false,
            message: "Profiler disabled.".to_string(),
        },
        ProfileCommandInput::Clear => ProfileCommandResult::Clear {
            message: "Profiler statistics cleared.".to_string(),
        },
        ProfileCommandInput::Invalid => ProfileCommandResult::Invalid {
            message: "Usage: profile [enable|disable|clear]".to_string(),
        },
    }
}

/// Apply a `profile` command directly to a simplifier and return user-facing text.
pub fn apply_profile_command(simplifier: &mut cas_engine::Simplifier, line: &str) -> String {
    match evaluate_profile_command_input(line) {
        ProfileCommandResult::ShowReport => simplifier.profiler.report(),
        ProfileCommandResult::SetEnabled { enabled, message } => {
            if enabled {
                simplifier.profiler.enable();
            } else {
                simplifier.profiler.disable();
            }
            message
        }
        ProfileCommandResult::Clear { message } => {
            simplifier.profiler.clear();
            message
        }
        ProfileCommandResult::Invalid { message } => message,
    }
}

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

/// Evaluate `vars` command lines using an explicit AST context.
pub fn evaluate_vars_command_lines_with_context(
    state: &SessionState,
    context: &cas_ast::Context,
) -> Vec<String> {
    evaluate_vars_command_lines(state, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    })
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

/// Evaluate `history` command lines using an explicit AST context.
pub fn evaluate_history_command_lines_with_context(
    state: &SessionState,
    context: &cas_ast::Context,
) -> Vec<String> {
    evaluate_history_command_lines(state, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    })
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

/// Format `show` command lines from a pre-computed inspection using explicit context.
pub fn format_show_history_command_lines_with_context<M>(
    inspection: &HistoryEntryInspection,
    context: &cas_ast::Context,
    mut metadata_lines: M,
) -> Vec<String>
where
    M: FnMut(&cas_ast::Context, &HistoryExprInspection) -> Vec<String>,
{
    let mut lines = format_history_entry_inspection_lines(inspection, |id| {
        format!("{}", cas_formatter::DisplayExpr { context, id })
    });
    if let HistoryEntryDetails::Expr(expr_info) = &inspection.details {
        lines.extend(metadata_lines(context, expr_info));
    }
    lines
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

/// Evaluate assignment command pieces and return formatted user-facing message.
pub fn evaluate_assignment_command_message_with_simplifier(
    state: &mut SessionState,
    simplifier: &mut cas_engine::Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String> {
    let output = evaluate_assignment_command(state, simplifier, name, expr_str, lazy)?;
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: output.expr
        }
    );
    Ok(format_assignment_command_output_message(&output, &rendered))
}

/// Evaluate `let ...` command tail and return formatted user-facing message.
pub fn evaluate_let_assignment_command_message_with_simplifier(
    state: &mut SessionState,
    simplifier: &mut cas_engine::Simplifier,
    input: &str,
) -> Result<String, String> {
    let output = evaluate_let_assignment_command(state, simplifier, input)?;
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: output.expr
        }
    );
    Ok(format_assignment_command_output_message(&output, &rendered))
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
        apply_profile_cache_command, apply_profile_command, evaluate_assignment_command,
        evaluate_assignment_command_message_with_simplifier, evaluate_clear_command_lines,
        evaluate_delete_history_command_message, evaluate_history_command_lines,
        evaluate_history_command_lines_with_context, evaluate_let_assignment_command,
        evaluate_let_assignment_command_message_with_simplifier, evaluate_vars_command_lines,
        evaluate_vars_command_lines_with_context, format_assignment_command_output_message,
        format_profile_cache_command_lines, format_show_history_command_lines,
        format_show_history_command_lines_with_context, parse_profile_command_input,
        ProfileCommandInput,
    };
    use crate::SessionState;

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
    fn evaluate_vars_command_lines_with_context_empty() {
        let state = SessionState::new();
        let ctx = cas_ast::Context::new();
        let lines = evaluate_vars_command_lines_with_context(&state, &ctx);
        assert_eq!(lines, vec!["No variables defined.".to_string()]);
    }

    #[test]
    fn evaluate_history_command_lines_with_context_empty() {
        let state = SessionState::new();
        let ctx = cas_ast::Context::new();
        let lines = evaluate_history_command_lines_with_context(&state, &ctx);
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
    fn format_show_history_command_lines_with_context_appends_metadata() {
        let mut ctx = cas_ast::Context::new();
        let parsed = cas_parser::parse("x + x", &mut ctx).expect("parse");
        let inspection = crate::HistoryEntryInspection {
            id: 1,
            type_str: "Expression".to_string(),
            raw_text: "x+x".to_string(),
            details: crate::HistoryEntryDetails::Expr(crate::HistoryExprInspection {
                parsed,
                resolved: None,
                simplified: None,
                required_conditions: Vec::new(),
                domain_warnings: Vec::new(),
                blocked_hints: Vec::new(),
            }),
        };
        let lines = format_show_history_command_lines_with_context(
            &inspection,
            &ctx,
            |_ctx, _expr_info| vec!["meta".to_string()],
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

    #[test]
    fn evaluate_assignment_command_message_with_simplifier_formats_success() {
        let mut state = SessionState::new();
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        let out = evaluate_assignment_command_message_with_simplifier(
            &mut state,
            &mut simplifier,
            "a",
            "x + x",
            true,
        )
        .expect("assign message");
        assert!(out.starts_with("a "));
    }

    #[test]
    fn evaluate_let_assignment_command_message_with_simplifier_formats_success() {
        let mut state = SessionState::new();
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        let out = evaluate_let_assignment_command_message_with_simplifier(
            &mut state,
            &mut simplifier,
            "a = x + x",
        )
        .expect("let message");
        assert!(out.starts_with("a "));
    }

    #[test]
    fn profile_command_parse_enable() {
        assert_eq!(
            parse_profile_command_input("profile enable"),
            ProfileCommandInput::Enable
        );
    }

    #[test]
    fn apply_profile_command_enable_and_disable_messages() {
        let mut simplifier = cas_engine::Simplifier::with_default_rules();
        assert_eq!(
            apply_profile_command(&mut simplifier, "profile enable"),
            "Profiler enabled."
        );
        assert_eq!(
            apply_profile_command(&mut simplifier, "profile disable"),
            "Profiler disabled."
        );
    }

    #[test]
    fn apply_profile_cache_command_status_and_format() {
        let mut engine = cas_engine::Engine::new();
        let result = apply_profile_cache_command(&mut engine, "cache status");
        let lines = format_profile_cache_command_lines(&result);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("Profile Cache: 0 profiles cached"));
    }
}

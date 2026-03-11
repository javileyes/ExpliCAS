use crate::SessionState;
#[allow(unused_imports)]
use cas_solver::session_api::{
    formatting::*, options::*, runtime::*, session_support::*, symbolic_commands::*, types::*,
};

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
fn evaluate_profile_cache_command_lines_status() {
    let mut engine = cas_solver::runtime::Engine::new();
    let lines = evaluate_profile_cache_command_lines(&mut engine, "cache status");
    assert!(lines.iter().any(|line| line.contains("Profile Cache:")));
}

#[test]
fn evaluate_solve_budget_command_message_returns_current_budget() {
    let mut state = SessionState::new();
    let message = evaluate_solve_budget_command_message(&mut state, "budget");
    assert!(message.contains("Solve budget"));
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
    let inspection = HistoryEntryInspection {
        id: 1,
        type_str: "Expression".to_string(),
        raw_text: "x+x".to_string(),
        details: HistoryEntryDetails::Expr(HistoryExprInspection {
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
    let inspection = HistoryEntryInspection {
        id: 1,
        type_str: "Expression".to_string(),
        raw_text: "x+x".to_string(),
        details: HistoryEntryDetails::Expr(HistoryExprInspection {
            parsed,
            resolved: None,
            simplified: None,
            required_conditions: Vec::new(),
            domain_warnings: Vec::new(),
            blocked_hints: Vec::new(),
        }),
    };
    let lines =
        format_show_history_command_lines_with_context(&inspection, &ctx, |_ctx, _expr_info| {
            vec!["meta".to_string()]
        });
    assert!(lines.iter().any(|line| line == "meta"));
}

#[test]
fn evaluate_assignment_command_success() {
    let mut state = SessionState::new();
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
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
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
    let err = evaluate_let_assignment_command(&mut state, &mut simplifier, "x + y")
        .expect_err("let parse error");
    assert!(err.contains("Usage:"));
}

#[test]
fn evaluate_assignment_command_message_with_simplifier_formats_success() {
    let mut state = SessionState::new();
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
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
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
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
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
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
    let mut engine = cas_solver::runtime::Engine::new();
    let result = apply_profile_cache_command(&mut engine, "cache status");
    let lines = format_profile_cache_command_lines(&result);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("Profile Cache: 0 profiles cached"));
}

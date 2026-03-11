use crate::solver_exports::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
    inspect_history_entry, inspect_history_entry_input, parse_history_entry_id,
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
use crate::SessionState;

#[test]
fn inspect_history_entry_reports_missing_id() {
    let mut state = SessionState::new();
    let mut engine = cas_solver::Engine::new();
    assert!(inspect_history_entry(&mut state, &mut engine, 999).is_none());
}

#[test]
fn inspect_history_entry_expr_contains_parsed() {
    let mut state = SessionState::new();
    let mut engine = cas_solver::Engine::new();
    let expr = cas_parser::parse("x + x", &mut engine.simplifier.context).expect("parse");
    let id = state.history_push(crate::EntryKind::Expr(expr), "x + x".to_string());

    let inspected = inspect_history_entry(&mut state, &mut engine, id).expect("entry");
    match inspected.details {
        HistoryEntryDetails::Expr(expr_info) => {
            assert_eq!(expr_info.parsed, expr);
        }
        HistoryEntryDetails::Eq { .. } => panic!("expected expr entry"),
    }
}

#[test]
fn parse_history_entry_id_accepts_hash_prefix() {
    let id = parse_history_entry_id("#12").expect("id parse");
    assert_eq!(id, 12);
}

#[test]
fn parse_history_entry_id_rejects_invalid_token() {
    let err = parse_history_entry_id("nope").expect_err("expected invalid id");
    assert_eq!(err, ParseHistoryEntryIdError::Invalid);
}

#[test]
fn inspect_history_entry_input_reports_not_found() {
    let mut state = SessionState::new();
    let mut engine = cas_solver::Engine::new();
    let err =
        inspect_history_entry_input(&mut state, &mut engine, "#3").expect_err("expected not-found");
    assert_eq!(err, InspectHistoryEntryInputError::NotFound { id: 3 });
}

#[test]
fn format_inspect_history_entry_error_message_invalid_id() {
    let msg = format_inspect_history_entry_error_message(&InspectHistoryEntryInputError::InvalidId);
    assert_eq!(msg, "Error: Invalid entry ID. Use 'show #N' or 'show N'.");
}

#[test]
fn format_inspect_history_entry_error_message_not_found() {
    let msg =
        format_inspect_history_entry_error_message(&InspectHistoryEntryInputError::NotFound {
            id: 9,
        });
    assert!(msg.contains("Entry #9 not found"));
    assert!(msg.contains("Use 'history'"));
}

#[test]
fn format_history_entry_inspection_lines_expr_includes_parsed() {
    let inspection = HistoryEntryInspection {
        id: 1,
        type_str: "Expression".to_string(),
        raw_text: "x + x".to_string(),
        details: HistoryEntryDetails::Expr(HistoryExprInspection {
            parsed: cas_ast::ExprId::from_raw(10),
            resolved: Some(cas_ast::ExprId::from_raw(11)),
            simplified: Some(cas_ast::ExprId::from_raw(12)),
            required_conditions: Vec::new(),
            domain_warnings: Vec::new(),
            blocked_hints: Vec::new(),
        }),
    };
    let lines = format_history_entry_inspection_lines(&inspection, |id| format!("E{}", id.index()));
    assert_eq!(lines[0], "Entry #1:");
    assert!(lines.iter().any(|line| line == "  Parsed:     E10"));
    assert!(lines.iter().any(|line| line == "  Resolved:   E11"));
    assert!(lines.iter().any(|line| line == "  Simplified: E12"));
}

#[test]
fn format_history_entry_inspection_lines_eq_includes_note() {
    let inspection = HistoryEntryInspection {
        id: 2,
        type_str: "Equation".to_string(),
        raw_text: "x = y".to_string(),
        details: HistoryEntryDetails::Eq {
            lhs: cas_ast::ExprId::from_raw(20),
            rhs: cas_ast::ExprId::from_raw(21),
        },
    };
    let lines = format_history_entry_inspection_lines(&inspection, |id| format!("E{}", id.index()));
    assert!(lines.iter().any(|line| line == "  LHS:        E20"));
    assert!(lines.iter().any(|line| line == "  RHS:        E21"));
    assert!(lines
        .iter()
        .any(|line| line.contains("When used as expression")));
}

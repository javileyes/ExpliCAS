use crate::{EntryId, EntryKind, SessionState};
use cas_ast::ExprId;

/// Errors when parsing a `show`-style history entry identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseHistoryEntryIdError {
    Invalid,
}

/// Errors when inspecting a history entry from command-style input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InspectHistoryEntryInputError {
    InvalidId,
    NotFound { id: EntryId },
}

/// Format inspection command errors as user-facing messages.
pub fn format_inspect_history_entry_error_message(error: &InspectHistoryEntryInputError) -> String {
    match error {
        InspectHistoryEntryInputError::InvalidId => {
            "Error: Invalid entry ID. Use 'show #N' or 'show N'.".to_string()
        }
        InspectHistoryEntryInputError::NotFound { id } => format!(
            "Error: Entry #{} not found.\nHint: Use 'history' to see available entries.",
            id
        ),
    }
}

/// High-level inspection payload for a history entry.
#[derive(Debug, Clone)]
pub struct HistoryEntryInspection {
    pub id: EntryId,
    pub type_str: String,
    pub raw_text: String,
    pub details: HistoryEntryDetails,
}

/// Per-kind inspection details.
#[derive(Debug, Clone)]
pub enum HistoryEntryDetails {
    Expr(HistoryExprInspection),
    Eq { lhs: ExprId, rhs: ExprId },
}

/// Format history-entry inspection lines using an expression renderer callback.
pub fn format_history_entry_inspection_lines<F>(
    inspection: &HistoryEntryInspection,
    mut render_expr: F,
) -> Vec<String>
where
    F: FnMut(ExprId) -> String,
{
    let mut lines = vec![
        format!("Entry #{}:", inspection.id),
        format!("  Type:       {}", inspection.type_str),
        format!("  Raw:        {}", inspection.raw_text),
    ];

    match &inspection.details {
        HistoryEntryDetails::Expr(expr_info) => {
            lines.push(format!("  Parsed:     {}", render_expr(expr_info.parsed)));
            if let Some(resolved) = expr_info.resolved {
                lines.push(format!("  Resolved:   {}", render_expr(resolved)));
            }
            if let Some(simplified) = expr_info.simplified {
                lines.push(format!("  Simplified: {}", render_expr(simplified)));
            }
        }
        HistoryEntryDetails::Eq { lhs, rhs } => {
            lines.push(format!("  LHS:        {}", render_expr(*lhs)));
            lines.push(format!("  RHS:        {}", render_expr(*rhs)));
            lines.push(String::new());
            lines.push("  Note: When used as expression, this becomes (LHS - RHS).".to_string());
        }
    }

    lines
}

/// Expression entry diagnostics and derived forms.
#[derive(Debug, Clone)]
pub struct HistoryExprInspection {
    pub parsed: ExprId,
    pub resolved: Option<ExprId>,
    pub simplified: Option<ExprId>,
    pub required_conditions: Vec<cas_solver::ImplicitCondition>,
    pub domain_warnings: Vec<cas_solver::DomainWarning>,
    pub blocked_hints: Vec<cas_solver::BlockedHint>,
}

/// Parse a `show`-style ID token (supports optional `#` prefix).
pub fn parse_history_entry_id(input: &str) -> Result<EntryId, ParseHistoryEntryIdError> {
    let token = input.trim().trim_start_matches('#');
    if token.is_empty() {
        return Err(ParseHistoryEntryIdError::Invalid);
    }
    token
        .parse::<EntryId>()
        .map_err(|_| ParseHistoryEntryIdError::Invalid)
}

/// Inspect a history entry and compute resolved/simplified forms plus metadata.
///
/// Returns `None` when `id` does not exist.
pub fn inspect_history_entry(
    state: &mut SessionState,
    engine: &mut cas_solver::Engine,
    id: EntryId,
) -> Option<HistoryEntryInspection> {
    let entry = state.history_get(id)?;
    let type_str = entry.type_str().to_string();
    let raw_text = entry.raw_text.clone();
    let kind = entry.kind.clone();

    let details = match kind {
        EntryKind::Expr(expr_id) => {
            let resolved_expr = state
                .resolve_state_refs(&mut engine.simplifier.context, expr_id)
                .ok()
                .filter(|resolved| *resolved != expr_id);
            let eval_req = cas_solver::EvalRequest {
                raw_input: raw_text.clone(),
                parsed: expr_id,
                action: cas_solver::EvalAction::Simplify,
                auto_store: false,
            };

            let expr_inspection = if let Ok(output) = engine.eval(state, eval_req) {
                let simplified = match output.result {
                    cas_solver::EvalResult::Expr(simplified)
                        if simplified != resolved_expr.unwrap_or(expr_id) =>
                    {
                        Some(simplified)
                    }
                    _ => None,
                };

                HistoryExprInspection {
                    parsed: expr_id,
                    resolved: resolved_expr,
                    simplified,
                    required_conditions: output.required_conditions,
                    domain_warnings: output.domain_warnings,
                    blocked_hints: cas_solver::blocked_hints_from_engine(&output.blocked_hints),
                }
            } else {
                let base = resolved_expr.unwrap_or(expr_id);
                let simplified = {
                    let (simplified, _steps) = engine.simplifier.simplify(base);
                    (simplified != base).then_some(simplified)
                };
                HistoryExprInspection {
                    parsed: expr_id,
                    resolved: resolved_expr,
                    simplified,
                    required_conditions: Vec::new(),
                    domain_warnings: Vec::new(),
                    blocked_hints: Vec::new(),
                }
            };

            HistoryEntryDetails::Expr(expr_inspection)
        }
        EntryKind::Eq { lhs, rhs } => HistoryEntryDetails::Eq { lhs, rhs },
    };

    Some(HistoryEntryInspection {
        id,
        type_str,
        raw_text,
        details,
    })
}

/// Parse and inspect a history entry in one call for command handlers.
pub fn inspect_history_entry_input(
    state: &mut SessionState,
    engine: &mut cas_solver::Engine,
    input: &str,
) -> Result<HistoryEntryInspection, InspectHistoryEntryInputError> {
    let id = parse_history_entry_id(input).map_err(|_| InspectHistoryEntryInputError::InvalidId)?;
    inspect_history_entry(state, engine, id).ok_or(InspectHistoryEntryInputError::NotFound { id })
}

#[cfg(test)]
mod tests {
    use super::{
        format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
        inspect_history_entry, inspect_history_entry_input, parse_history_entry_id,
        HistoryEntryDetails, InspectHistoryEntryInputError, ParseHistoryEntryIdError,
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
        let err = inspect_history_entry_input(&mut state, &mut engine, "#3")
            .expect_err("expected not-found");
        assert_eq!(err, InspectHistoryEntryInputError::NotFound { id: 3 });
    }

    #[test]
    fn format_inspect_history_entry_error_message_invalid_id() {
        let msg =
            format_inspect_history_entry_error_message(&InspectHistoryEntryInputError::InvalidId);
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
        let inspection = super::HistoryEntryInspection {
            id: 1,
            type_str: "Expression".to_string(),
            raw_text: "x + x".to_string(),
            details: super::HistoryEntryDetails::Expr(super::HistoryExprInspection {
                parsed: cas_ast::ExprId::from_raw(10),
                resolved: Some(cas_ast::ExprId::from_raw(11)),
                simplified: Some(cas_ast::ExprId::from_raw(12)),
                required_conditions: Vec::new(),
                domain_warnings: Vec::new(),
                blocked_hints: Vec::new(),
            }),
        };
        let lines =
            format_history_entry_inspection_lines(&inspection, |id| format!("E{}", id.index()));
        assert_eq!(lines[0], "Entry #1:");
        assert!(lines.iter().any(|line| line == "  Parsed:     E10"));
        assert!(lines.iter().any(|line| line == "  Resolved:   E11"));
        assert!(lines.iter().any(|line| line == "  Simplified: E12"));
    }

    #[test]
    fn format_history_entry_inspection_lines_eq_includes_note() {
        let inspection = super::HistoryEntryInspection {
            id: 2,
            type_str: "Equation".to_string(),
            raw_text: "x = y".to_string(),
            details: super::HistoryEntryDetails::Eq {
                lhs: cas_ast::ExprId::from_raw(20),
                rhs: cas_ast::ExprId::from_raw(21),
            },
        };
        let lines =
            format_history_entry_inspection_lines(&inspection, |id| format!("E{}", id.index()));
        assert!(lines.iter().any(|line| line == "  LHS:        E20"));
        assert!(lines.iter().any(|line| line == "  RHS:        E21"));
        assert!(lines
            .iter()
            .any(|line| line.contains("When used as expression")));
    }
}

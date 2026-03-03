use crate::{EntryId, EntryKind, SessionState};

use crate::inspect_types::{
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};

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
                let output_view = cas_solver::eval_output_view(&output);
                let simplified = match output_view.result {
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
                    required_conditions: output_view.required_conditions,
                    domain_warnings: output_view.domain_warnings,
                    blocked_hints: output_view.blocked_hints,
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

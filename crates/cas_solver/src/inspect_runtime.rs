use cas_ast::ExprId;
use cas_session_core::types::EntryId;

use crate::{
    eval_output_view, Engine, EvalAction, EvalOutput, EvalRequest, EvalResult, HistoryEntryDetails,
    HistoryEntryInspection, HistoryEntryKindRaw, HistoryExprInspection,
    InspectHistoryEntryInputError,
};

/// Raw history entry payload needed for `show`/inspection flows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryInspectEntryRaw {
    pub id: EntryId,
    pub type_str: String,
    pub raw_text: String,
    pub kind: HistoryEntryKindRaw,
}

/// Context needed to inspect a history entry with optional eval metadata.
pub trait InspectHistoryContext {
    fn history_entry_raw(&self, id: EntryId) -> Option<HistoryInspectEntryRaw>;

    fn resolve_state_refs_for_inspect(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, String>;

    fn eval_for_inspect(
        &mut self,
        engine: &mut Engine,
        request: EvalRequest,
    ) -> Result<EvalOutput, String>;
}

/// Inspect a history entry and compute resolved/simplified forms plus metadata.
///
/// Returns `None` when `id` does not exist.
pub fn inspect_history_entry<C: InspectHistoryContext>(
    context: &mut C,
    engine: &mut Engine,
    id: EntryId,
) -> Option<HistoryEntryInspection> {
    let entry = context.history_entry_raw(id)?;

    let details = match entry.kind {
        HistoryEntryKindRaw::Expr(expr_id) => {
            let resolved_expr = context
                .resolve_state_refs_for_inspect(&mut engine.simplifier.context, expr_id)
                .ok()
                .filter(|resolved| *resolved != expr_id);

            let eval_req = EvalRequest {
                raw_input: entry.raw_text.clone(),
                parsed: expr_id,
                action: EvalAction::Simplify,
                auto_store: false,
            };

            let expr_inspection = if let Ok(output) = context.eval_for_inspect(engine, eval_req) {
                let output_view = eval_output_view(&output);
                let simplified = match output_view.result {
                    EvalResult::Expr(simplified)
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
        HistoryEntryKindRaw::Eq { lhs, rhs } => HistoryEntryDetails::Eq { lhs, rhs },
    };

    Some(HistoryEntryInspection {
        id: entry.id,
        type_str: entry.type_str,
        raw_text: entry.raw_text,
        details,
    })
}

/// Parse and inspect a history entry in one call for command handlers.
pub fn inspect_history_entry_input<C: InspectHistoryContext>(
    context: &mut C,
    engine: &mut Engine,
    input: &str,
) -> Result<HistoryEntryInspection, InspectHistoryEntryInputError> {
    let id = crate::parse_history_entry_id(input)
        .map_err(|_| InspectHistoryEntryInputError::InvalidId)?;
    inspect_history_entry(context, engine, id).ok_or(InspectHistoryEntryInputError::NotFound { id })
}

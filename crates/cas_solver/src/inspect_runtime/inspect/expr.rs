mod eval;
mod fallback;

use cas_ast::ExprId;

use super::super::InspectHistoryContext;
use crate::{Engine, HistoryExprInspection};

pub(super) fn inspect_history_expr<C: InspectHistoryContext>(
    context: &mut C,
    engine: &mut Engine,
    raw_text: &str,
    expr_id: ExprId,
) -> HistoryExprInspection {
    let resolved_expr = context
        .resolve_state_refs_for_inspect(&mut engine.simplifier.context, expr_id)
        .ok()
        .filter(|resolved| *resolved != expr_id);

    eval::inspect_history_expr_via_eval(context, engine, raw_text, expr_id, resolved_expr)
        .unwrap_or_else(|| fallback::inspect_history_expr_fallback(engine, expr_id, resolved_expr))
}

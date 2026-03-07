use cas_ast::ExprId;

use crate::{Engine, HistoryExprInspection};

pub(super) fn inspect_history_expr_fallback(
    engine: &mut Engine,
    expr_id: ExprId,
    resolved_expr: Option<ExprId>,
) -> HistoryExprInspection {
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
}

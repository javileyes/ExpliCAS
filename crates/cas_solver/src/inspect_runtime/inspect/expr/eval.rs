use cas_ast::ExprId;

use crate::{
    eval_output_view, inspect_runtime::InspectHistoryContext, Engine, EvalAction, EvalRequest,
    EvalResult, HistoryExprInspection,
};

pub(super) fn inspect_history_expr_via_eval<C: InspectHistoryContext>(
    context: &mut C,
    engine: &mut Engine,
    raw_text: &str,
    expr_id: ExprId,
    resolved_expr: Option<ExprId>,
) -> Option<HistoryExprInspection> {
    let eval_req = EvalRequest {
        raw_input: raw_text.to_string(),
        parsed: expr_id,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = context.eval_for_inspect(engine, eval_req).ok()?;
    let output_view = eval_output_view(&output);
    let simplified = match output_view.result {
        EvalResult::Expr(simplified) if simplified != resolved_expr.unwrap_or(expr_id) => {
            Some(simplified)
        }
        _ => None,
    };

    Some(HistoryExprInspection {
        parsed: expr_id,
        resolved: resolved_expr,
        simplified,
        required_conditions: output_view.required_conditions,
        domain_warnings: output_view.domain_warnings,
        blocked_hints: output_view.blocked_hints,
    })
}

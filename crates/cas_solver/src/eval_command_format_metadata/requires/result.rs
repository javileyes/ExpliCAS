pub(super) fn eval_result_expr(result: &crate::EvalResult) -> Option<cas_ast::ExprId> {
    match result {
        crate::EvalResult::Expr(expr_id) => Some(*expr_id),
        _ => None,
    }
}

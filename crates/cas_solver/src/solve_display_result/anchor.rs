use cas_ast::ExprId;

pub fn requires_result_expr_anchor(result: &crate::EvalResult, resolved: ExprId) -> ExprId {
    match result {
        crate::EvalResult::Expr(expr) => *expr,
        crate::EvalResult::Set(values) => *values.first().unwrap_or(&resolved),
        _ => resolved,
    }
}

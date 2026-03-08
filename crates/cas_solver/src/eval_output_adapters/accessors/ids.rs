/// Get stored history id, if any.
pub fn stored_id_from_eval_output(output: &crate::EvalOutput) -> Option<u64> {
    output.stored_id
}

/// Get parsed expression id.
pub fn parsed_expr_from_eval_output(output: &crate::EvalOutput) -> cas_ast::ExprId {
    output.parsed
}

/// Get resolved expression id.
pub fn resolved_expr_from_eval_output(output: &crate::EvalOutput) -> cas_ast::ExprId {
    output.resolved
}

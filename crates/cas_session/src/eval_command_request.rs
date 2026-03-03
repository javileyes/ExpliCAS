pub(crate) fn build_simplify_eval_request_from_statement(
    ctx: &mut cas_ast::Context,
    raw_input: &str,
    stmt: cas_parser::Statement,
    auto_store: bool,
) -> cas_solver::EvalRequest {
    cas_solver::EvalRequest {
        raw_input: raw_input.to_string(),
        parsed: crate::input_parse_common::statement_to_expr_id(ctx, stmt),
        action: cas_solver::EvalAction::Simplify,
        auto_store,
    }
}

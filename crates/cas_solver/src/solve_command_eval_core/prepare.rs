use super::PreparedSolveEvalRequest;

pub(crate) fn prepare_solve_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
    auto_store: bool,
) -> Result<PreparedSolveEvalRequest, crate::SolvePrepareError> {
    let (parsed_expr, original_equation, var) =
        crate::prepare_solve_expr_and_var(ctx, input, explicit_var)?;

    Ok(PreparedSolveEvalRequest {
        raw_input: input.to_string(),
        parsed_expr,
        auto_store,
        var,
        original_equation,
    })
}

pub(super) fn format_eval_result_text(
    ctx: &cas_ast::Context,
    result: &crate::EvalResult,
) -> String {
    match result {
        crate::EvalResult::Expr(expr) => crate::display_expr_or_poly(ctx, *expr),
        crate::EvalResult::Set(values) if !values.is_empty() => format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: values[0]
            }
        ),
        crate::EvalResult::Bool(value) => value.to_string(),
        _ => "(no result)".to_string(),
    }
}

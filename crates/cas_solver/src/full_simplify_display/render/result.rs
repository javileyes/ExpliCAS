pub(super) fn append_full_simplify_result_line(
    lines: &mut Vec<String>,
    ctx: &mut cas_ast::Context,
    simplified_expr: cas_ast::ExprId,
    style_prefs: &cas_formatter::StylePreferences,
) {
    lines.push(format!(
        "Result: {}",
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExprStyled::new(ctx, simplified_expr, style_prefs)
        ))
    ));
}

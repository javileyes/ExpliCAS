pub(super) fn prepare_full_simplify_render(
    ctx: &mut cas_ast::Context,
    expr_input: &str,
    resolved_expr: cas_ast::ExprId,
) -> (Vec<String>, cas_formatter::StylePreferences) {
    let style_signals = cas_formatter::ParseStyleSignals::from_input_string(expr_input);
    let style_prefs = cas_formatter::StylePreferences::from_expression_with_signals(
        ctx,
        resolved_expr,
        Some(&style_signals),
    );

    let lines = vec![format!(
        "Parsed: {}",
        cas_formatter::DisplayExpr {
            context: &*ctx,
            id: resolved_expr
        }
    )];

    (lines, style_prefs)
}

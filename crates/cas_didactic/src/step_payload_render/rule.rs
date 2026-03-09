use cas_ast::ExprId;

pub(super) fn render_local_rule_latex(
    context: &cas_ast::Context,
    before_expr: ExprId,
    after_expr: ExprId,
) -> String {
    let local_before_colored =
        render_highlighted_expr(context, before_expr, cas_formatter::HighlightColor::Red);
    let local_after_colored =
        render_highlighted_expr(context, after_expr, cas_formatter::HighlightColor::Green);
    format!(
        "{} \\rightarrow {}",
        local_before_colored, local_after_colored
    )
}

fn render_highlighted_expr(
    context: &cas_ast::Context,
    expr_id: ExprId,
    color: cas_formatter::HighlightColor,
) -> String {
    let mut config = cas_formatter::HighlightConfig::new();
    config.add(expr_id, color);
    cas_formatter::LaTeXExprHighlighted {
        context,
        id: expr_id,
        highlights: &config,
    }
    .to_latex()
}

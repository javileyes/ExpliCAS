use cas_ast::{ExprId, ExprPath};

pub(super) fn render_step_path_latex(
    context: &cas_ast::Context,
    expr_id: ExprId,
    expr_path: ExprPath,
    color: cas_formatter::HighlightColor,
) -> String {
    let mut config = cas_formatter::PathHighlightConfig::new();
    config.add(expr_path, color);
    cas_formatter::PathHighlightedLatexRenderer {
        context,
        id: expr_id,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    }
    .to_latex()
}

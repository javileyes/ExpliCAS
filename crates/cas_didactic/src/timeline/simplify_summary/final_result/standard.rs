use cas_ast::{Context, ExprId};
use cas_formatter::{
    DisplayContext, PathHighlightConfig, PathHighlightedLatexRenderer, StylePreferences,
};

pub(super) fn render_standard_final_result_html(
    context: &Context,
    final_result_expr: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    let empty_config = PathHighlightConfig::new();
    let final_expr = PathHighlightedLatexRenderer {
        context,
        id: final_result_expr,
        path_highlights: &empty_config,
        hints: Some(display_hints),
        style_prefs: Some(style_prefs),
    }
    .to_latex();

    let mut html = String::from(
        r#"        </div>
        <div class="final-result">
            \(\textbf{Final Result:}\)
            \["#,
    );
    html.push_str(&final_expr);
    html.push_str(
        r#"\]
        </div>
"#,
    );
    html
}

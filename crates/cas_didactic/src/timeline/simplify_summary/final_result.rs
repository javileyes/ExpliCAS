use cas_ast::{Context, ExprId};
use cas_formatter::{
    html_escape, DisplayContext, PathHighlightConfig, PathHighlightedLatexRenderer,
    StylePreferences,
};

pub(super) fn render_timeline_final_result_html(
    context: &Context,
    final_result_expr: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    if let Some(poly_text) =
        cas_math::poly_store::try_render_poly_result(context, final_result_expr)
    {
        return render_polynomial_final_result_html(&poly_text);
    }

    render_standard_final_result_html(context, final_result_expr, display_hints, style_prefs)
}

fn render_polynomial_final_result_html(poly_text: &str) -> String {
    let term_count = poly_text.matches('+').count() + 1;
    let mut html = String::from(
        r#"        </div>
        <div class="final-result">
            <strong>🧮 Final Result</strong> <span class="poly-badge">"#,
    );
    html.push_str(&format!("Polynomial: {} terms", term_count));
    html.push_str(
        r#"</span>
            <pre class="poly-output">"#,
    );
    html.push_str(&html_escape(poly_text));
    html.push_str(
        r#"</pre>
        </div>
"#,
    );
    html
}

fn render_standard_final_result_html(
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

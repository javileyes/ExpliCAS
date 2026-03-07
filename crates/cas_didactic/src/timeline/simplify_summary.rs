use cas_ast::{Context, ExprId};
use cas_formatter::{
    html_escape, DisplayContext, PathHighlightConfig, PathHighlightedLatexRenderer,
    StylePreferences,
};
use cas_solver::{render_conditions_normalized, ImplicitCondition};

pub(super) fn render_timeline_final_result_html(
    context: &Context,
    final_result_expr: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    if let Some(poly_text) =
        cas_math::poly_store::try_render_poly_result(context, final_result_expr)
    {
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
        html.push_str(&html_escape(&poly_text));
        html.push_str(
            r#"</pre>
        </div>
"#,
        );
        return html;
    }

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

pub(super) fn render_timeline_global_requires_html(
    context: &mut Context,
    global_requires: &[ImplicitCondition],
) -> String {
    if global_requires.is_empty() {
        return String::new();
    }

    let requires_messages = render_conditions_normalized(context, global_requires);
    if requires_messages.is_empty() {
        return String::new();
    }

    let escaped: Vec<String> = requires_messages.iter().map(|s| html_escape(s)).collect();
    format!(
        r#"        <div class="global-requires">
            <strong>ℹ️ Requires:</strong> {}
        </div>
"#,
        escaped.join(", ")
    )
}

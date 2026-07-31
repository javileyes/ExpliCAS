use cas_formatter::html_escape;

pub(super) fn render_enriched_substep(sub: &crate::didactic::SubStep) -> String {
    // The template drops both sides inside a MathJax `\[...\]` block, so prefer
    // the declared LaTeX; `before_expr`/`after_expr` are plain display text.
    let before_math = sub.before_latex.as_deref().unwrap_or(&sub.before_expr);
    let after_math = sub.after_latex.as_deref().unwrap_or(&sub.after_expr);
    let math_html = if before_math.is_empty() {
        String::new()
    } else {
        super::super::super::render_template::render_timeline_asset!(
            "simplify_render/substep_math.html",
            &[
                ("__BEFORE_EXPR__", before_math),
                ("__AFTER_EXPR__", after_math),
            ],
        )
    };
    let description_html = html_escape(&sub.description);

    super::super::super::render_template::render_timeline_asset!(
        "simplify_render/enriched_substep.html",
        &[
            ("__DESCRIPTION__", description_html.as_str()),
            ("__MATH_HTML__", math_html.as_str()),
        ],
    )
}

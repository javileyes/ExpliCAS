use cas_formatter::html_escape;

pub(super) fn render_enriched_substep(sub: &crate::didactic::SubStep) -> String {
    let math_html = if sub.before_expr.is_empty() {
        String::new()
    } else {
        super::super::super::render_template::render_static_template(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/timeline/simplify_render/substep_math.html"
            )),
            &[
                ("__BEFORE_EXPR__", sub.before_expr.as_str()),
                ("__AFTER_EXPR__", sub.after_expr.as_str()),
            ],
        )
    };
    let description_html = html_escape(&sub.description);

    super::super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/enriched_substep.html"
        )),
        &[
            ("__DESCRIPTION__", description_html.as_str()),
            ("__MATH_HTML__", math_html.as_str()),
        ],
    )
}

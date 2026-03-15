pub(super) fn render_global_requires_html(requires_messages: &[String]) -> String {
    if requires_messages.is_empty() {
        return String::new();
    }

    super::super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/global_requires.html"
        )),
        &[("__REQUIRES_MESSAGES__", &requires_messages.join(", "))],
    )
}

pub(super) fn render_global_requires_html(requires_messages: &[String]) -> String {
    if requires_messages.is_empty() {
        return String::new();
    }

    super::super::super::render_template::render_timeline_asset!(
        "simplify_render/global_requires.html",
        &[("__REQUIRES_MESSAGES__", &requires_messages.join(", "))],
    )
}

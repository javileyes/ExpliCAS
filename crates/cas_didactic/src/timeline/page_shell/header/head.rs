use cas_formatter::html_escape;

pub(super) fn render_timeline_page_head(
    page_title_prefix: &str,
    title: &str,
    common_css: &str,
    extra_css: &str,
) -> String {
    let page_title_prefix_html = html_escape(page_title_prefix);
    let title_html = html_escape(title);
    super::super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/page_shell/header/head.html"
        )),
        &[
            ("__PAGE_TITLE_PREFIX__", page_title_prefix_html.as_str()),
            ("__TITLE__", title_html.as_str()),
            ("__COMMON_CSS__", common_css),
            ("__EXTRA_CSS__", extra_css),
        ],
    )
}

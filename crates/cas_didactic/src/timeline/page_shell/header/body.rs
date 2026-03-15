use cas_formatter::html_escape;

pub(super) fn render_timeline_page_body_intro(
    heading: &str,
    subtitle_html: &str,
    original_label: &str,
    original_latex: &str,
    theme_toggle_script: &str,
) -> String {
    let heading_html = html_escape(heading);
    let original_label_html = html_escape(original_label);
    super::super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/page_shell/header/body_intro.html"
        )),
        &[
            ("__THEME_TOGGLE_SCRIPT__", theme_toggle_script),
            ("__HEADING__", heading_html.as_str()),
            ("__SUBTITLE_HTML__", subtitle_html),
            ("__ORIGINAL_LABEL__", original_label_html.as_str()),
            ("__ORIGINAL_LATEX__", original_latex),
        ],
    )
}

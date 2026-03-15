pub(super) fn render_timeline_page_footer(footer_html: &str, extra_script: Option<&str>) -> String {
    let extra_script_html = extra_script.unwrap_or_default();
    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/page_shell/footer.html"
        )),
        &[
            ("__EXTRA_SCRIPT__", extra_script_html),
            ("__FOOTER_HTML__", footer_html),
        ],
    )
}

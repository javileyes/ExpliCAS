pub(super) fn render_timeline_page_footer(footer_html: &str, extra_script: Option<&str>) -> String {
    let extra_script_html = extra_script.unwrap_or_default();
    format!(
        r#"{}    <footer>
        {}
    </footer>
</body>
</html>"#,
        extra_script_html, footer_html
    )
}

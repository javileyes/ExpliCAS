mod footer;
mod header;
mod theme_toggle_script;

pub(super) struct TimelinePageShell<'a> {
    pub page_title_prefix: &'a str,
    pub title: &'a str,
    pub heading: &'a str,
    pub subtitle_html: &'a str,
    pub original_label: &'a str,
    pub original_latex: &'a str,
    pub extra_css: &'a str,
}

pub(super) fn render_timeline_page_header(shell: TimelinePageShell<'_>) -> String {
    header::render_timeline_page_header(shell)
}

pub(super) fn render_timeline_page_footer(footer_html: &str, extra_script: Option<&str>) -> String {
    footer::render_timeline_page_footer(footer_html, extra_script)
}

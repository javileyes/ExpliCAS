pub(super) const TIMELINE_CLOSING_HTML: &str = super::prepare::TIMELINE_CLOSING_HTML;

pub(super) fn open_timeline_html() -> String {
    crate::timeline::render_template::timeline_asset!("simplify_render/timeline_open.html")
        .to_string()
}

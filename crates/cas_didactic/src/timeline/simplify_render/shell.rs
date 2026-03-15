pub(super) const TIMELINE_CLOSING_HTML: &str = super::prepare::TIMELINE_CLOSING_HTML;

pub(super) fn open_timeline_html() -> String {
    String::from(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/timeline/simplify_render/timeline_open.html"
    )))
}

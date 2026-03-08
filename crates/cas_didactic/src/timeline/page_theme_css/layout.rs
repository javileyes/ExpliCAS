mod page;
mod theme_toggle;
mod timeline;

pub(super) fn layout_css() -> String {
    [
        &page::page_css(),
        &theme_toggle::theme_toggle_css(),
        timeline::TIMELINE_CSS,
    ]
    .concat()
}

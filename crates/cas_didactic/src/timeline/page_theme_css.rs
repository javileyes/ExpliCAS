mod base;
mod components;
mod layout;
mod theme;

pub(super) fn common_timeline_page_css() -> String {
    [
        base::BASE_CSS,
        theme::THEME_VARIABLES_CSS,
        layout::LAYOUT_CSS,
        &components::components_css(),
    ]
    .concat()
}

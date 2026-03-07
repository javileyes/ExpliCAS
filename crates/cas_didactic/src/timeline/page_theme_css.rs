mod base;
mod components;
mod layout;
mod theme;

pub(super) fn common_timeline_page_css() -> String {
    [
        base::BASE_CSS,
        &theme::theme_variables_css(),
        &layout::layout_css(),
        &components::components_css(),
    ]
    .concat()
}

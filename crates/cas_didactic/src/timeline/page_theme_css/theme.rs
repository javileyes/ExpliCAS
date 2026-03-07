mod dark;
mod light;
mod shared;

pub(super) fn theme_variables_css() -> String {
    shared::join_css_sections([
        dark::dark_theme_variables_css(),
        light::light_theme_variables_css(),
    ])
}

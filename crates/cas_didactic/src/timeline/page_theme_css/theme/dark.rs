mod accent;
mod semantic;
mod surface;
use super::shared::join_css_sections;

pub(super) fn dark_theme_variables_css() -> String {
    join_css_sections([
        accent::ACCENT_CSS.to_string(),
        surface::SURFACE_CSS.to_string(),
        semantic::SEMANTIC_CSS.to_string(),
    ])
}

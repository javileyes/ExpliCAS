mod content;
mod surface;

pub(super) fn page_css() -> String {
    [surface::SURFACE_CSS, content::CONTENT_CSS].concat()
}

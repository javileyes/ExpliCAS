mod poly;
mod requires;

pub(super) fn summary_css() -> String {
    [poly::SUMMARY_POLY_CSS, requires::SUMMARY_REQUIRES_CSS].concat()
}

mod requires;
mod warning;

pub(super) fn details_domain_css() -> String {
    [
        warning::DETAILS_DOMAIN_WARNING_CSS,
        requires::DETAILS_DOMAIN_REQUIRES_CSS,
    ]
    .concat()
}

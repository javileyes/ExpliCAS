mod content;
mod details;
mod summary;

pub(super) fn simplify_timeline_extra_css() -> String {
    [
        content::CONTENT_CSS,
        details::DETAILS_CSS,
        summary::SUMMARY_CSS,
    ]
    .concat()
}

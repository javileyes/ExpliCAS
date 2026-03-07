mod content;
mod details;
mod summary;

pub(super) fn simplify_timeline_extra_css() -> String {
    [
        content::content_css(),
        details::details_css(),
        summary::summary_css(),
    ]
    .concat()
}

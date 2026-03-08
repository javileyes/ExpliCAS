mod details;
mod item;

pub(super) fn substeps_css() -> String {
    [details::SUBSTEPS_DETAILS_CSS, item::SUBSTEP_ITEM_CSS].concat()
}

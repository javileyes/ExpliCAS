mod container;
mod item;
mod toggle;

pub(super) fn substeps_css() -> String {
    [
        toggle::SUBSTEPS_TOGGLE_CSS,
        container::SUBSTEPS_CONTAINER_CSS,
        item::SUBSTEP_ITEM_CSS,
    ]
    .concat()
}

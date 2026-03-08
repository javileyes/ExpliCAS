mod domain;
mod item;
mod shell;

pub(super) fn details_css() -> String {
    [
        shell::DETAILS_SHELL_CSS,
        item::DETAILS_ITEM_CSS,
        &domain::details_domain_css(),
    ]
    .concat()
}

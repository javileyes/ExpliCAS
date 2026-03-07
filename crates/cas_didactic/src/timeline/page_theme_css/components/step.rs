mod card;
mod labels;
mod warning;

pub(super) fn step_css() -> String {
    [
        card::STEP_CARD_CSS,
        labels::STEP_LABELS_CSS,
        warning::WARNING_BOX_CSS,
    ]
    .concat()
}

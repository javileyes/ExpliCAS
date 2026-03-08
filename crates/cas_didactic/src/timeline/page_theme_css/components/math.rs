mod blocks;
mod rule;

pub(super) fn math_css() -> String {
    [blocks::MATH_BLOCKS_CSS, rule::RULE_BOX_CSS].concat()
}

mod math;
mod rule;
mod shell;

pub(super) fn content_css() -> String {
    [
        shell::CONTENT_SHELL_CSS,
        math::CONTENT_MATH_CSS,
        rule::CONTENT_RULE_CSS,
    ]
    .concat()
}

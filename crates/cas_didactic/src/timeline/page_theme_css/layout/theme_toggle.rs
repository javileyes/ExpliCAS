mod shell;
mod switch;

pub(super) fn theme_toggle_css() -> String {
    [
        shell::THEME_TOGGLE_SHELL_CSS,
        switch::THEME_TOGGLE_SWITCH_CSS,
    ]
    .concat()
}

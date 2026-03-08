mod body;
mod head;

use super::super::page_theme_css::common_timeline_page_css;
use super::theme_toggle_script::THEME_TOGGLE_SCRIPT;
use super::TimelinePageShell;

pub(super) fn render_timeline_page_header(shell: TimelinePageShell<'_>) -> String {
    let common_css = common_timeline_page_css();
    format!(
        "{}{}",
        head::render_timeline_page_head(
            shell.page_title_prefix,
            shell.title,
            &common_css,
            shell.extra_css,
        ),
        body::render_timeline_page_body_intro(
            shell.heading,
            shell.subtitle_html,
            shell.original_label,
            shell.original_latex,
            THEME_TOGGLE_SCRIPT,
        ),
    )
}

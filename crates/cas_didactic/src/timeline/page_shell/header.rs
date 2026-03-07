use super::super::page_theme_css::common_timeline_page_css;
use super::theme_toggle_script::THEME_TOGGLE_SCRIPT;
use super::TimelinePageShell;
use cas_formatter::html_escape;

pub(super) fn render_timeline_page_header(shell: TimelinePageShell<'_>) -> String {
    let common_css = common_timeline_page_css();
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}: {}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>{}{}
    </style>
</head>
<body>
    <div class="theme-toggle">
        <span>🌙</span>
        <label class="toggle-switch">
            <input type="checkbox" id="themeToggle" onchange="toggleTheme()">
            <span class="toggle-slider"></span>
        </label>
        <span>☀️</span>
    </div>
    {}
    <div class="container">
        <h1>{}</h1>
        <p class="subtitle">{}</p>
        <div class="original">
            \(\textbf{{{}}}\)
            \[{}\]
        </div>
"#,
        html_escape(shell.page_title_prefix),
        html_escape(shell.title),
        common_css,
        shell.extra_css,
        THEME_TOGGLE_SCRIPT,
        html_escape(shell.heading),
        shell.subtitle_html,
        html_escape(shell.original_label),
        shell.original_latex,
    )
}

use super::page_theme_css::COMMON_TIMELINE_PAGE_CSS;
use cas_formatter::html_escape;

pub(super) struct TimelinePageShell<'a> {
    pub page_title_prefix: &'a str,
    pub title: &'a str,
    pub heading: &'a str,
    pub subtitle_html: &'a str,
    pub original_label: &'a str,
    pub original_latex: &'a str,
    pub extra_css: &'a str,
}

pub(super) fn render_timeline_page_header(shell: TimelinePageShell<'_>) -> String {
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
    <script>
        function toggleTheme() {{
            document.documentElement.classList.toggle('light');
            localStorage.setItem('theme', document.documentElement.classList.contains('light') ? 'light' : 'dark');
        }}
        if (localStorage.getItem('theme') === 'light') {{
            document.documentElement.classList.add('light');
            document.getElementById('themeToggle').checked = true;
        }}
    </script>
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
        COMMON_TIMELINE_PAGE_CSS,
        shell.extra_css,
        html_escape(shell.heading),
        shell.subtitle_html,
        html_escape(shell.original_label),
        shell.original_latex,
    )
}

pub(super) fn render_timeline_page_footer(footer_html: &str, extra_script: Option<&str>) -> String {
    let extra_script_html = extra_script.unwrap_or_default();
    format!(
        r#"{}    <footer>
        {}
    </footer>
</body>
</html>"#,
        extra_script_html, footer_html
    )
}

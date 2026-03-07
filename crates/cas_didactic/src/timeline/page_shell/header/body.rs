use cas_formatter::html_escape;

pub(super) fn render_timeline_page_body_intro(
    heading: &str,
    subtitle_html: &str,
    original_label: &str,
    original_latex: &str,
    theme_toggle_script: &str,
) -> String {
    format!(
        r#"<body>
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
        theme_toggle_script,
        html_escape(heading),
        subtitle_html,
        html_escape(original_label),
        original_latex,
    )
}

use cas_formatter::html_escape;

pub(super) fn render_timeline_page_head(
    page_title_prefix: &str,
    title: &str,
    common_css: &str,
    extra_css: &str,
) -> String {
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
"#,
        html_escape(page_title_prefix),
        html_escape(title),
        common_css,
        extra_css,
    )
}

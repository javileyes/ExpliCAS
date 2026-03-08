pub(super) fn render_global_requires_html(requires_messages: &[String]) -> String {
    if requires_messages.is_empty() {
        return String::new();
    }

    format!(
        r#"        <div class="global-requires">
            <strong>ℹ️ Requires:</strong> {}
        </div>
"#,
        requires_messages.join(", ")
    )
}

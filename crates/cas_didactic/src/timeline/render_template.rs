pub(super) fn render_static_template(template: &str, replacements: &[(&str, &str)]) -> String {
    replacements
        .iter()
        .fold(template.to_string(), |acc, (key, value)| {
            acc.replace(key, value)
        })
}

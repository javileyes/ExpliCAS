pub(super) fn strip_text_wrappers(mut value: String) -> String {
    while let Some(start) = value.find("\\text{") {
        let Some(end) = value[start + 6..].find('}') else {
            break;
        };
        let content = &value[start + 6..start + 6 + end];
        value = format!(
            "{}{}{}",
            &value[..start],
            content,
            &value[start + 7 + end..]
        );
    }
    value
}

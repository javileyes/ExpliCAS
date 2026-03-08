pub(super) fn render_single_conditional_case_line(case_lines: &[String]) -> Option<String> {
    let [single] = case_lines else {
        return None;
    };

    single
        .find(r" & \text{if}")
        .map(|index| single[..index].to_string())
}

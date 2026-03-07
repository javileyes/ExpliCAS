pub(super) fn render_conditional_case_lines(case_lines: &[String]) -> String {
    if case_lines.len() == 1 {
        let single = &case_lines[0];
        if let Some(index) = single.find(r" & \text{if}") {
            return single[..index].to_string();
        }
    }

    format!(
        r"\begin{{cases}} {} \end{{cases}}",
        case_lines.join(r" \\ ")
    )
}

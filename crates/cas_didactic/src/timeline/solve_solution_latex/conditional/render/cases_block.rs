pub(super) fn render_conditional_cases_block(case_lines: &[String]) -> String {
    format!(
        r"\begin{{cases}} {} \end{{cases}}",
        case_lines.join(r" \\ ")
    )
}

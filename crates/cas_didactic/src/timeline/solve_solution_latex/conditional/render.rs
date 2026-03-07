mod cases_block;
mod single;

pub(super) fn render_conditional_case_lines(case_lines: &[String]) -> String {
    single::render_single_conditional_case_line(case_lines)
        .unwrap_or_else(|| cases_block::render_conditional_cases_block(case_lines))
}

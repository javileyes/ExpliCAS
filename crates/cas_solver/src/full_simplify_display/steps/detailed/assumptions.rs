pub(super) fn push_detailed_assumption_lines(lines: &mut Vec<String>, step: &crate::Step) {
    for assumption_line in crate::format_displayable_assumption_lines_for_step(step) {
        lines.push(format!("   {}", assumption_line));
    }
}

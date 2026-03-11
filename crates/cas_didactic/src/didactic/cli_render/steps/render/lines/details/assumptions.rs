use crate::cas_solver::Step;

pub(super) fn render_assumption_lines(step: &Step) -> Vec<String> {
    crate::cas_solver::format_displayable_assumption_lines_for_step(step)
        .into_iter()
        .map(|line| format!("   {}", line))
        .collect()
}

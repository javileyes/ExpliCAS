use cas_solver::Step;

pub(super) fn render_engine_substeps_lines(step: &Step) -> Vec<String> {
    let mut lines = Vec::new();
    if !step.substeps().is_empty() {
        for substep in step.substeps() {
            lines.push(format!("   [{}]", substep.title));
            for line in &substep.lines {
                lines.push(format!("      • {}", line));
            }
        }
    }
    lines
}

pub(super) fn render_assumption_lines(step: &Step) -> Vec<String> {
    cas_solver::format_displayable_assumption_lines_for_step(step)
        .into_iter()
        .map(|line| format!("   {}", line))
        .collect()
}

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

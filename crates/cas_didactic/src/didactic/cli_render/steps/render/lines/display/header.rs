use cas_solver::Step;

pub(super) fn render_step_header(step_count: usize, step: &Step) -> String {
    format!("{}. {}  [{}]", step_count, step.description, step.rule_name)
}

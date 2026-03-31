use crate::runtime::Step;

pub(super) fn render_step_header(step_count: usize, step: &Step) -> String {
    let description = crate::didactic::visible_step_description(&step.description);
    format!("{}. {}", step_count, description)
}

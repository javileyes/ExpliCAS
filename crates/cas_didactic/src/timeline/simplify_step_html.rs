mod frame;
mod sections;

use crate::runtime::Step;
use cas_formatter::html_escape;

use super::simplify_highlights::TimelineRenderedStepMath;

pub(super) fn render_timeline_step_html(
    step_number: usize,
    step: &Step,
    rendered_step_math: &TimelineRenderedStepMath,
    sub_steps_html: &str,
    rule_substeps_html: &str,
    domain_html: &str,
) -> String {
    let step_title = html_escape(&step.rule_name);
    let before_html = sections::render_before_section(&rendered_step_math.global_before);
    let rule_html = sections::render_rule_section(step, &rendered_step_math.local_change_latex);
    let after_html = sections::render_after_section(&rendered_step_math.global_after);

    frame::render_step_frame(
        step_number,
        &step_title,
        &before_html,
        sub_steps_html,
        &rule_html,
        rule_substeps_html,
        &after_html,
        domain_html,
    )
}

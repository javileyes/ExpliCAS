use crate::runtime::Step;
use cas_formatter::html_escape;

pub(super) fn render_timeline_domain_assumptions_html(step: &Step) -> String {
    let grouped_lines = crate::runtime::format_displayable_assumption_lines_grouped_for_step(step);
    if grouped_lines.is_empty() {
        return String::new();
    }

    let parts: Vec<String> = grouped_lines.iter().map(|line| html_escape(line)).collect();
    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/domain_assumptions.html"
        )),
        &[("__GROUPED_LINES__", &parts.join("<br/>"))],
    )
}

use cas_formatter::html_escape;
use cas_solver::Step;

pub(super) fn render_timeline_domain_assumptions_html(step: &Step) -> String {
    let grouped_lines = cas_solver::format_displayable_assumption_lines_grouped_for_step(step);
    if grouped_lines.is_empty() {
        return String::new();
    }

    let parts: Vec<String> = grouped_lines.iter().map(|line| html_escape(line)).collect();
    format!(
        r#"                    <div class="domain-assumptions">{}</div>
"#,
        parts.join("<br/>")
    )
}

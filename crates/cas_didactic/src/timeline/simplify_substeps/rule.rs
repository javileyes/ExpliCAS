use crate::runtime::Step;
use cas_formatter::html_escape;

pub(super) fn render_timeline_rule_substeps_html(step: &Step) -> String {
    if step.substeps().is_empty() {
        return String::new();
    }

    let mut details_html = String::from(
        r#"<details class="substeps-details" open>
                    <summary>Pasos didácticos</summary>
                    <div class="substeps-content">"#,
    );
    for substep in step.substeps() {
        details_html.push_str(&format!(
            r#"<div class="substep">
                            <strong>[{}]</strong>"#,
            html_escape(&substep.title)
        ));
        for line in &substep.lines {
            details_html.push_str(&format!(
                r#"<div class="substep-line">• {}</div>"#,
                html_escape(line)
            ));
        }
        details_html.push_str("</div>");
    }
    details_html.push_str("</div></details>");
    details_html
}

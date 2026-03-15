use crate::runtime::Step;
use cas_formatter::html_escape;

pub(super) fn render_timeline_rule_substeps_html(step: &Step) -> String {
    if step.substeps().is_empty() {
        return String::new();
    }

    let mut content_html = String::new();
    for substep in step.substeps() {
        let mut lines_html = String::new();
        for line in &substep.lines {
            let line_html = html_escape(line);
            lines_html.push_str(&super::super::render_template::render_static_template(
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/assets/timeline/simplify_render/rule_substep_line.html"
                )),
                &[("__LINE__", line_html.as_str())],
            ));
        }
        let title_html = html_escape(&substep.title);
        content_html.push_str(&super::super::render_template::render_static_template(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/timeline/simplify_render/rule_substep_item.html"
            )),
            &[
                ("__TITLE__", title_html.as_str()),
                ("__LINES__", lines_html.as_str()),
            ],
        ));
    }

    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/substeps_details.html"
        )),
        &[
            ("__OPEN_ATTR__", " open"),
            ("__SUMMARY__", "Pasos didácticos"),
            ("__CONTENT__", content_html.as_str()),
        ],
    )
}

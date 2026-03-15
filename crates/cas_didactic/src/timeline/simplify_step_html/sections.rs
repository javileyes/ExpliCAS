use crate::runtime::Step;

pub(super) fn render_before_section(global_before: &str) -> String {
    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/before_section.html"
        )),
        &[("__GLOBAL_BEFORE__", global_before)],
    )
}

pub(super) fn render_rule_section(step: &Step, local_change_latex: &str) -> String {
    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/rule_section.html"
        )),
        &[
            ("__STEP_DESCRIPTION__", step.description.as_str()),
            ("__LOCAL_CHANGE_LATEX__", local_change_latex),
        ],
    )
}

pub(super) fn render_after_section(global_after: &str) -> String {
    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/after_section.html"
        )),
        &[("__GLOBAL_AFTER__", global_after)],
    )
}

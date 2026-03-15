#[allow(clippy::too_many_arguments)]
pub(super) fn render_step_frame(
    step_number: usize,
    step_title: &str,
    before_html: &str,
    sub_steps_html: &str,
    rule_html: &str,
    rule_substeps_html: &str,
    after_html: &str,
    domain_html: &str,
) -> String {
    let step_number_text = step_number.to_string();
    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/step_frame.html"
        )),
        &[
            ("__STEP_NUMBER__", step_number_text.as_str()),
            ("__STEP_TITLE__", step_title),
            ("__BEFORE_HTML__", before_html),
            ("__SUB_STEPS_HTML__", sub_steps_html),
            ("__RULE_HTML__", rule_html),
            ("__RULE_SUBSTEPS_HTML__", rule_substeps_html),
            ("__AFTER_HTML__", after_html),
            ("__DOMAIN_HTML__", domain_html),
        ],
    )
}

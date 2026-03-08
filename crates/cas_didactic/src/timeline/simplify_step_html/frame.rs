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
    format!(
        r#"            <div class="step">
                <div class="step-number">{}</div>
                <div class="step-content">
                    <h3>{}</h3>
                    {}{}{}{}{}{}                </div>
            </div>
"#,
        step_number,
        step_title,
        before_html,
        sub_steps_html,
        rule_html,
        rule_substeps_html,
        after_html,
        domain_html
    )
}

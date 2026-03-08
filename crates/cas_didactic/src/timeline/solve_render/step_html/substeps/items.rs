use cas_formatter::html_escape;

pub(super) fn render_substep_item_html(
    step_number: usize,
    substep_number: usize,
    description: &str,
    sub_eq_latex: &str,
) -> String {
    format!(
        r#"                <div class="substep">
                    <div class="substep-number">Step {}.{}</div>
                    <div class="substep-description">{}</div>
                    <div class="substep-equation">
                        \[{}\]
                    </div>
                </div>
"#,
        step_number,
        substep_number,
        html_escape(description),
        sub_eq_latex
    )
}

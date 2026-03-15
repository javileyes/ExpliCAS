use cas_formatter::html_escape;

pub(super) fn render_substep_item_html(
    step_number: usize,
    substep_number: usize,
    description: &str,
    sub_eq_latex: &str,
) -> String {
    let step_number_text = step_number.to_string();
    let substep_number_text = substep_number.to_string();
    let description_html = html_escape(description);
    super::super::super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/solve_render/substep_item.html"
        )),
        &[
            ("__STEP_NUMBER__", step_number_text.as_str()),
            ("__SUBSTEP_NUMBER__", substep_number_text.as_str()),
            ("__DESCRIPTION__", description_html.as_str()),
            ("__EQUATION_LATEX__", sub_eq_latex),
        ],
    )
}

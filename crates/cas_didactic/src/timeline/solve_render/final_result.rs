use cas_formatter::html_escape;

pub(super) fn render_solve_final_result_html(var: &str, solution_latex: &str) -> String {
    let var_html = html_escape(var);
    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/solve_render/final_result.html"
        )),
        &[
            ("__VAR__", var_html.as_str()),
            ("__SOLUTION_LATEX__", solution_latex),
        ],
    )
}

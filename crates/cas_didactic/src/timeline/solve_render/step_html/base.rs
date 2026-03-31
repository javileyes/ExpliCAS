use crate::runtime::SolveStep;
use cas_formatter::html_escape;

pub(super) const STEP_CLOSE_HTML: &str =
    crate::timeline::render_template::timeline_asset!("solve_render/step_close.html");

pub(super) fn render_solve_step_open_html(
    step_number: usize,
    step: &SolveStep,
    eq_latex: &str,
) -> String {
    let step_number_text = step_number.to_string();
    let description_html = html_escape(&step.description);
    super::super::super::render_template::render_timeline_asset!(
        "solve_render/step_open.html",
        &[
            ("__STEP_NUMBER__", step_number_text.as_str()),
            ("__DESCRIPTION__", description_html.as_str()),
            ("__EQUATION_LATEX__", eq_latex),
        ],
    )
}

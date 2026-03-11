use crate::runtime::SolveStep;
use cas_formatter::html_escape;

pub(super) const STEP_CLOSE_HTML: &str = "        </div>\n";

pub(super) fn render_solve_step_open_html(
    step_number: usize,
    step: &SolveStep,
    eq_latex: &str,
) -> String {
    format!(
        r#"        <div class="step">
            <div class="step-number">Step {}</div>
            <div class="description">{}</div>
            <div class="equation">
                \[{}\]
            </div>
"#,
        step_number,
        html_escape(&step.description),
        eq_latex
    )
}

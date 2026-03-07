use super::render_equation_latex;
use cas_ast::Context;
use cas_formatter::html_escape;
use cas_solver::SolveStep;

pub(super) fn render_solve_step_html(
    context: &Context,
    step_number: usize,
    step: &SolveStep,
) -> String {
    let eq_latex = render_equation_latex(context, &step.equation_after);

    let mut html = format!(
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
    );

    if !step.substeps.is_empty() {
        let substep_id = format!("substeps-{}", step_number);
        html.push_str(&format!(
            r#"            <div class="substeps-toggle" onclick="toggleSubsteps('{}')">
                <span class="arrow">▶</span>
                <span>Show derivation ({} steps)</span>
            </div>
            <div id="{}" class="substeps-container">
"#,
            substep_id,
            step.substeps.len(),
            substep_id
        ));

        for (j, substep) in step.substeps.iter().enumerate() {
            let sub_eq_latex = render_equation_latex(context, &substep.equation_after);
            html.push_str(&format!(
                r#"                <div class="substep">
                    <div class="substep-number">Step {}.{}</div>
                    <div class="substep-description">{}</div>
                    <div class="substep-equation">
                        \[{}\]
                    </div>
                </div>
"#,
                step_number,
                j + 1,
                html_escape(&substep.description),
                sub_eq_latex
            ));
        }

        html.push_str("            </div>\n");
    }

    html.push_str("        </div>\n");
    html
}

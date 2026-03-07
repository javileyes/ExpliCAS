mod items;
mod toggle;

use super::super::render_equation_latex;
use cas_ast::Context;
use cas_solver::SolveStep;

pub(super) fn render_solve_substeps_html(
    context: &Context,
    step_number: usize,
    step: &SolveStep,
) -> String {
    if step.substeps.is_empty() {
        return String::new();
    }

    let substep_id = format!("substeps-{}", step_number);
    let mut html = toggle::render_substeps_toggle_html(&substep_id, step.substeps.len());

    for (j, substep) in step.substeps.iter().enumerate() {
        let sub_eq_latex = render_equation_latex(context, &substep.equation_after);
        html.push_str(&items::render_substep_item_html(
            step_number,
            j + 1,
            &substep.description,
            &sub_eq_latex,
        ));
    }

    html.push_str("            </div>\n");
    html
}

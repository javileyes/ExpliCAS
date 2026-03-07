mod base;
mod substeps;

use super::render_equation_latex;
use cas_ast::Context;
use cas_solver::SolveStep;

pub(super) fn render_solve_step_html(
    context: &Context,
    step_number: usize,
    step: &SolveStep,
) -> String {
    let eq_latex = render_equation_latex(context, &step.equation_after);
    let mut html = base::render_solve_step_open_html(step_number, step, &eq_latex);
    html.push_str(&substeps::render_solve_substeps_html(
        context,
        step_number,
        step,
    ));
    html.push_str(base::STEP_CLOSE_HTML);
    html
}

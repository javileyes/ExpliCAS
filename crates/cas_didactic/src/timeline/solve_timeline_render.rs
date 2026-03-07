use super::solve_render::{render_solve_final_result_html, render_solve_step_html};
use super::solve_solution_latex::render_solution_set_to_latex;
use cas_ast::{Context, SolutionSet};
use cas_solver::SolveStep;

pub(super) fn render_solve_timeline_body(
    context: &Context,
    steps: &[SolveStep],
    solution_set: &SolutionSet,
    var: &str,
) -> String {
    let mut html = String::from("        <div class=\"timeline\">\n");

    for (i, step) in steps.iter().enumerate() {
        let step_number = i + 1;
        html.push_str(&render_solve_step_html(context, step_number, step));
    }

    let solution_latex = render_solution_set_to_latex(context, solution_set);
    html.push_str(&render_solve_final_result_html(var, &solution_latex));
    html
}
